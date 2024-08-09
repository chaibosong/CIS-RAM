import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, num_classes):
        super(MLP, self).__init__()
        model = []
        model.append(nn.Linear(128, 512))
        model.append(nn.LeakyReLU())
        model.append(nn.Linear(512, 256))
        model.append(nn.LeakyReLU())
        model.append(nn.Linear(256, 128))
        model.append(nn.LeakyReLU())
        model.append(nn.Linear(128, num_classes))
        # model.append(nn.Softmax())
        model.append(nn.Softmax(dim=1))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x

