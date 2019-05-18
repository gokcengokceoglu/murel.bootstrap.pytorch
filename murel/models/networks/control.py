import torch.nn as nn


class ControlModule(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ControlModule, self).__init__()
        self.rnncell = nn.RNNCell(input_size=input_size, hidden_size=hidden_size, nonlinearity='tanh')

    def forward(self, input, hidden):
        hidden_next = self.rnncell(input, hidden)
        return hidden_next
