import torch
import torch.nn as nn
import torch.nn.functional as F


class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(7, 3)
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x):
        x = x.view(-1, 7)
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)