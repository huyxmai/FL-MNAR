import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from BiasCorr.BiasCorr_preprocess import g_s_train, binary_cross_entropy

def BiasCorr_train_FA(
        beta, gamma, rho, sigma, training_set, optimizer, pi_2k_c, epochs, J, R, device='cpu'
):

    # Construct pi_c
    pi_c = Variable(
        torch.cat((
            torch.log(1 - torch.exp(pi_2k_c)), pi_2k_c
        )),
        requires_grad=True)

    X_train = training_set[:, :-2]

    optimizer_psi = torch.optim.SGD([pi_c], lr=0.85)

    M = F.gumbel_softmax(logits=pi_c, tau=1, hard=True, dim=0)
    m = torch.argmax(M, dim=0).reshape(1, -1)
    psi = torch.matmul(torch.Tensor([[0, 1]]), M)

    # Recalculate s' based on new assignment
    X_train_s = torch.mul(X_train, psi.detach())
    not_select = (training_set[:, -1] != 1)

    sel_var = g_s_train(
        X_train_sel=X_train_s, X_train_sel_u=X_train_s[not_select],
        s=training_set[:, -1], spec='LR', tol=0.075, lr=0.01
    )

    training_set[not_select, -1] = torch.Tensor([sel_var])

    for epoch in range(epochs):

        # Shuffle training set
        shuffled_indices = torch.randperm(training_set.shape[0])
        shuffled_X = X_train[shuffled_indices]
        shuffled_y = training_set[shuffled_indices, -2]
        shuffled_s = training_set[shuffled_indices, -1]

        shuffled_X_s = torch.mul(shuffled_X, psi)

        # Get shuffled X_p
        shuffled_X_p = torch.cat((
            shuffled_X[:, :J], torch.zeros((shuffled_X.shape[0], shuffled_X.shape[1] - J))
        ), 1)

        # Add intercepts
        shuffled_X_p = torch.cat((
            torch.ones((shuffled_X_p.shape[0], 1)), shuffled_X_p
        ), 1)
        shuffled_X_s = torch.cat((
            torch.ones((shuffled_X.shape[0], 1)), shuffled_X_s
        ), 1)

        # Train
        random_draws = torch.randn(shuffled_X_p.shape[0], R)
        Sigma = sigma.clamp(0, 2)
        Rho = rho.clamp(-0.99, 0.99)
        nu = gamma / ((1 - Rho ** 2) ** ( 1 /2))
        tau = Rho / ((1 - Rho ** 2) ** ( 1 /2))

        optimizer.zero_grad()

        beta_product = torch.inner(beta, shuffled_X_p)

        beta_product = torch.t(beta_product).repeat(1, R)
        nu_product = torch.inner(nu, shuffled_X_s)
        nu_product = torch.t(nu_product).repeat(1, R)

        obs_pred1 = torch.exp(beta_product + Sigma * random_draws) / \
                    (1 + torch.exp(beta_product + Sigma * random_draws))
        obs_pred0 = 1 - obs_pred1

        value = (1 - shuffled_s.reshape(-1, 1).repeat(1, R) \
                 + shuffled_s.reshape(-1, 1).repeat(1, R) \
                 * (shuffled_y.reshape(-1, 1).repeat(1, R) * obs_pred1 \
                    + (1 - shuffled_y.reshape(-1, 1).repeat(1, R)) * obs_pred0)) \
                * torch.distributions.Normal(loc=0, scale=1).cdf(
            (2 * shuffled_s.reshape(-1, 1).repeat(1, R) - 1) * (nu_product + tau * random_draws))

        NLL = torch.log(torch.sum(value, 1) / R)
        loss = -1 * torch.mean(NLL)

        loss.backward(retain_graph=True)
        optimizer.step()

        if epoch + 1 == epochs:
            optimizer_psi.zero_grad()

            # Compute L_p
            random_epsilons = torch.randn(1, shuffled_X_s.shape[0])
            gamma_hat = gamma.detach()
            rho_hat = rho.detach()

            nu = gamma_hat / ((1 - rho_hat ** 2) ** (1 / 2))
            tau = rho_hat / ((1 - rho_hat ** 2) ** (1 / 2))
            nu_product = torch.matmul(nu, shuffled_X_s.T)

            sel1 = (1 + torch.erf(nu_product + tau * random_epsilons)) / 2
            sel0 = 1 - sel1

            value = shuffled_s.reshape(1, -1) * torch.log(sel1) + (1 - shuffled_s.reshape(1, -1)) * torch.log(sel0)
            L_p = -1 * torch.mean(value)

            L_p.backward(retain_graph=True)

            optimizer_psi.step()

    new_pi_2k_c = pi_c[1, :]
    new_pi_2k_c = new_pi_2k_c.reshape(1, -1)

    temp_pi_2k_c = torch.clone(new_pi_2k_c).detach()
    temp_pi_2k_c[:, :J] = torch.ones((1, J)) * np.log(0.9999)

    return beta, temp_pi_2k_c


def BiasCorr_test(beta, testing_set, J):
    test_y = testing_set[:, -2]

    # Get X_p
    X_test_p = torch.cat((
        torch.ones((testing_set.shape[0], 1)), testing_set[:, :J]
    ), 1)

    beta_hat = beta[0]
    beta_hat = beta_hat[:, :(J + 1)]

    with torch.no_grad():
        test_predictions = np.exp(np.matmul(beta_hat, X_test_p.T)) / (
                1 + np.exp(np.matmul(beta_hat,
                                     X_test_p.T)))

        loss = binary_cross_entropy(test_predictions.reshape(-1, 1), test_y.reshape(-1, 1))

        test_predictions = np.round_(test_predictions)
        test_predictions = np.reshape(test_predictions, (X_test_p.shape[0],)).detach().cpu().numpy().astype(int)
        correct = [test_y[i] == test_predictions[i] for i in range(len(test_y))].count(True)
        test_acc = correct / X_test_p.shape[0]

    return loss, test_acc