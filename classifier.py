# Copyright (c) Alibaba Group
import torch
import torch.nn as nn

class SeA(nn.Module):
    """
    Build a linear model for sea
    """

    def __init__(self, input_dim, output_dim):
        super(SeA, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        return self.fc(x)