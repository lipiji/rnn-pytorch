import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1

        self.W_x_ifoc = nn.Parameter(torch.Tensor(4 * self.hidden_size, self.input_size))
        self.W_h_ifoc = nn.Parameter(torch.Tensor(4 * self.hidden_size, self.hidden_size))
        self.b_ifoc = nn.Parameter(torch.Tensor(4 * self.hidden_size))

        self.init_parameters()

    def init_parameters(self):
        nn.init.orthogonal_(self.W_x_ifoc)
        nn.init.orthogonal_(self.W_h_ifoc)
        self.b_ifoc.data.fill_(0)

    def forward(self, X, hidden, mask = None):
        X_4ifoc = F.linear(X, self.W_x_ifoc, self.b_ifoc)
        def recurrence(x_4ifoc, hidden, m = None):
            pre_h, pre_c = hidden
            pre_h = pre_h.view(-1, self.hidden_size)
            pre_c = pre_c.view(-1, self.hidden_size)
            
            ifoc_preact = x_4ifoc + F.linear(pre_c, self.W_h_ifoc)

            x4i, x4f, x4o, x4c = ifoc_preact.chunk(4, 1)
            i = torch.sigmoid(x4i)
            f = torch.sigmoid(x4f)
            o = torch.sigmoid(x4o)
            c = f * pre_c + i * torch.tanh(x4c)
            h = o * torch.tanh(c)
            
            if m is not None:
                m = m.view(-1, 1)
                h = h * m + pre_h * (1 - m)
                c = c * m + pre_c * (1 - m)

            return h, c

        hiddens = []
        steps = range(X.size(0))
        for i in steps:
            if mask is not None:
                hidden = recurrence(X_4ifoc[i], hidden, mask[i])
            else:
                hidden = recurrence(X_4ifoc[i], hidden)
            hiddens.append(hidden[0])

        hiddens = torch.cat(hiddens, 0).view(X.size(0), *hiddens[0].size())
        return hiddens, hidden



