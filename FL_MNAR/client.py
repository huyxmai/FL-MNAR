import flwr as fl
import torch
from typing import Dict
from flwr.common import Scalar
import numpy as np
from BiasCorr.BiasCorr_train_test import BiasCorr_train_FA, BiasCorr_test

class FullAdultClient(fl.client.NumPyClient):
    def __init__(self,
                 cid,
                 training_set,
                 K,
                 J,
                 E,
                 T_s,
                 alpha,
                 device) -> None:
        super().__init__()

        self.cid = cid
        self.training_set = training_set
        self.device = device

        self.K = K
        self.J = J
        self.E = E
        self.T_s = T_s
        self.alpha = alpha

    def set_parameters(self, parameters):

        self.beta = torch.Tensor(parameters[0])[:, :self.training_set.shape[1] - 1]
        self.gamma = torch.zeros((1, self.training_set.shape[1] - 1))
        self.rho = torch.Tensor([0.01])
        self.sigma = torch.Tensor([0.01])

        self.pi_2k_original = torch.Tensor(parameters[2])
        self.pi_2k_c = torch.Tensor(parameters[2])[self.cid, :].reshape(1, -1)

    def get_parameters(self, config: Dict[str, Scalar]):

        return [
            np.concatenate((
                self.beta.detach().cpu().numpy(), np.zeros((1, self.K + 1 - self.beta.shape[1]))
            ), axis=1),
            self.pi_2k_c.detach().cpu().numpy(),
            self.pi_2k_original.detach().cpu().numpy(),
            self.cid
        ]

    def fit(self, parameters, config):

        R = 200

        self.set_parameters(parameters)

        # set requires_grad = True
        self.beta.requires_grad = True
        self.gamma.requires_grad = True
        self.rho.requires_grad = True
        self.sigma.requires_grad = True

        optimizer = torch.optim.SGD([self.beta, self.gamma, self.rho, self.sigma], lr=self.alpha, momentum=0.9)

        if config["server_round"] % self.T_s == 0:
            self.beta, self.pi_2k_c = BiasCorr_train_FA(
                beta=self.beta, gamma=self.gamma, rho=self.rho, sigma=self.sigma,
                training_set=self.training_set, optimizer=optimizer,
                pi_2k_c=self.pi_2k_c, epochs=self.E, J=self.J, R=R, device=self.device
            )
        else:
            print("==no change==")
            self.beta, _ = BiasCorr_train_FA(
                beta=self.beta, gamma=self.gamma, rho=self.rho, sigma=self.sigma,
                training_set=self.training_set, optimizer=optimizer,
                pi_2k_c=self.pi_2k_c, epochs=self.E, J=self.J, R=R, device=self.device
            )

        return self.get_parameters({}), len(self.training_set), {}

    def evaluate(self, parameters, config: Dict[str, Scalar]):

        self.set_parameters(parameters)
        loss, accuracy = BiasCorr_test(beta=self.beta, training_set=self.training_set, J=self.J)

        return float(loss), len(self.training_set), {'accuracy': accuracy}

def generate_full_client_fn(training_sets, K, J, E, T_s, alpha, device):
    def full_client_fn(cid: str):
        return FullAdultClient(
            cid=int(cid), training_set=training_sets[int(cid)], K=K, J=J, E=E, T_s=T_s, alpha=alpha, device=device
        )

    return full_client_fn