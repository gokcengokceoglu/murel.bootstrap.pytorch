import torch.nn as nn
import torch



class ControlModule(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ControlModule, self).__init__()
        self.rnncell = nn.RNNCell(input_size=input_size, hidden_size=hidden_size, nonlinearity='tanh')

    def forward(self, input, hidden):

        hidden_next = self.rnncell(input, hidden)
        hidden_next_check = torch.isnan(hidden_next)

        if hidden_next_check.sum() > 0 :
            print("This actually corresponds to c. I don't know why I put this. ")

        return hidden_next


