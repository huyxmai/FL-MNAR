import torch
import torch.nn as nn
import torch.nn.functional as F

class LogReg(nn.Module):

    def __init__(self, in_feat):
        super(LogReg, self).__init__()
        self.layer1 = nn.Linear(in_feat, 1)
        torch.nn.init.zeros_(self.layer1.weight)
        torch.nn.init.zeros_(self.layer1.bias)

    def forward(self, x):
        x = self.layer1(x)
        out = torch.sigmoid(x)
        return out

class MultilayerPerceptron(nn.Module):

    def __init__(self, in_feat):
        super(MultilayerPerceptron, self).__init__()

        # Layers
        self.layer1 = nn.Linear(in_feat, 5)
        self.layer2 = nn.Linear(5, 2)
        self.layer3 = nn.Linear(2, 1)

        torch.nn.init.normal_(self.layer1.weight)
        torch.nn.init.normal_(self.layer1.bias)
        torch.nn.init.normal_(self.layer2.weight)
        torch.nn.init.normal_(self.layer2.bias)
        torch.nn.init.zeros_(self.layer3.weight)
        torch.nn.init.zeros_(self.layer3.bias)

    def forward(self, x):
        x1 = self.layer1(x)
        x1 = F.relu(x1)
        x2 = self.layer2(x1)
        x2 = F.relu(x2)
        x3 = self.layer3(x2)
        out = torch.sigmoid(x3)
        return out