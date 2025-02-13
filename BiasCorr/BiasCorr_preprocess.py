import numpy as np
import torch
from torch.autograd import Variable
from scipy.stats import norm
from BiasCorr.models import LogReg, MultilayerPerceptron

def binary_cross_entropy(output, target):
    loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)
    return -1 * torch.mean(loss)

def g_s_train(X_train_sel, X_train_sel_u, s, spec, tol, lr):

    if spec == 'LR':
        clf = LogReg(in_feat=X_train_sel.shape[1])
        optimizer = torch.optim.SGD(clf.parameters(), lr=lr, weight_decay=0)
        it = 0

        while True:
            it += 1

            # Shuffle training set
            shuffled_indices = torch.randperm(X_train_sel.shape[0])
            shuffled_train_selection = X_train_sel[shuffled_indices]
            shuffled_s = s[shuffled_indices]

            optimizer.zero_grad()
            output = clf(shuffled_train_selection)
            loss = binary_cross_entropy(output, shuffled_s.reshape(-1, 1))

            # Check convergence criteria
            if it == 1:
                prev_loss = loss.cpu().detach().numpy()
            else:
                if (np.abs(loss.cpu().detach().numpy() - prev_loss) * 100) / prev_loss < tol:
                    break
                else:
                    prev_loss = loss.cpu().detach().numpy()

            loss.backward()
            optimizer.step()

            with torch.no_grad():

                train_sel_u_predictions = clf(X_train_sel_u).cpu().detach().numpy()
                last_observed_u = train_sel_u_predictions.flatten()

    elif spec == 'Probit':
        X_train_sel = torch.cat((
            torch.ones((X_train_sel.shape[0], 1)), X_train_sel
        ), 1)
        X_train_sel_u = torch.cat((
            torch.ones((X_train_sel_u.shape[0], 1)), X_train_sel_u
        ), 1)
        val_epsilons = np.random.normal(X_train_sel_u.shape[0])

        gamma = Variable(torch.zeros((1, X_train_sel.shape[1])), requires_grad=True)
        rho = Variable(torch.Tensor([0.01]), requires_grad=True)

        optimizer = torch.optim.SGD([gamma, rho], lr=lr, weight_decay=0)

        prev_loss = 0
        it = 0

        while True:
            it += 1

            # Shuffle training set
            shuffled_indices = torch.randperm(X_train_sel.shape[0])
            shuffled_train_selection = X_train_sel[shuffled_indices]
            shuffled_s = s[shuffled_indices]

            random_epsilons = torch.randn(1, X_train_sel.shape[0])

            Rho = rho.clamp(-0.99, 0.99)

            nu = gamma / ((1 - Rho ** 2) ** (1 / 2))
            tau = Rho / ((1 - Rho ** 2) ** (1 / 2))

            optimizer.zero_grad()

            nu_product = torch.matmul(nu, shuffled_train_selection.T)

            sel1 = (1 + torch.special.erf(nu_product + tau * random_epsilons)) / 2
            sel0 = 1 - sel1

            value = shuffled_s.reshape(1, -1) * torch.log(sel1) + (1 - shuffled_s.reshape(1, -1)) * torch.log(sel0)
            loss = -1 * torch.mean(value)

            # Check convergence criteria
            if it == 1:
                prev_loss = loss.cpu().detach().numpy()
            else:
                if (np.abs(loss.cpu().detach().numpy() - prev_loss) * 100) / prev_loss < tol:
                    break
                else:
                    prev_loss = loss.cpu().detach().numpy()

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                train_sel_u_predictions = norm.cdf \
                    (np.matmul(nu.cpu().detach().numpy() ,X_train_sel_u.T) + tau.cpu().detach().numpy() * val_epsilons)
                last_observed_u = train_sel_u_predictions.reshape(-1, 1).flatten()

    sel_var = np.mean(last_observed_u)

    return sel_var

def g_y_train(X_train_s, X_train_u, train_y_s, spec, tol, lr):
    if spec == 'LR':
        clf = LogReg(in_feat=X_train_s.shape[1])
    elif spec == 'MLP':
        clf = MultilayerPerceptron(in_feat=X_train_s.shape[1])

    optimizer = torch.optim.SGD(clf.parameters(), lr=lr, weight_decay=0)

    it = 0
    while True:
        it += 1

        # Shuffle data
        shuffled_indices = torch.randperm(X_train_s.shape[0])
        shuffled_train_prediction = X_train_s[shuffled_indices]
        shuffled_y = train_y_s[shuffled_indices]

        optimizer.zero_grad()
        output = clf(shuffled_train_prediction)
        loss = binary_cross_entropy(output, shuffled_y.reshape(-1, 1))

        # Check convergence criteria
        if it == 1:
            prev_loss = loss.cpu().detach().numpy()
        else:
            if (np.abs(loss.cpu().detach().numpy() - prev_loss) * 100) / prev_loss < tol:
                break
            else:
                prev_loss = loss.cpu().detach().numpy()

        loss.backward()
        optimizer.step()

    # Get pseudolabels
    with torch.no_grad():
        train_u_predictions_clf = clf(X_train_u)

        train_u_predictions = np.round_(train_u_predictions_clf.cpu().detach().numpy())
        pseudolabels = np.reshape(train_u_predictions, (X_train_u.shape[0],)).astype(int)

    return pseudolabels