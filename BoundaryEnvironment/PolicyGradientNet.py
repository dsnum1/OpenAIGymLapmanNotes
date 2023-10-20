import torch
import torch.nn as nn

class PolicyGradientNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyGradientNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )


    def forward(self,x):
        return self.net(torch.FloatTensor(x))
    