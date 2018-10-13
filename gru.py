import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_x_rz = nn.Parameter(torch.Tensor(2 * self.hidden_size, self.input_size))
        self.W_h_rz = nn.Parameter(torch.Tensor(2 * self.hidden_size, self.hidden_size))
        self.b_rz = nn.Parameter(torch.Tensor(2 * self.hidden_size))
        self.W_xh = nn.Parameter(torch.Tensor(self.hidden_size, self.input_size))
        self.W_hh = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_h = nn.Parameter(torch.Tensor(self.hidden_size))

        self.init_parameters()

    def init_parameters(self):
        nn.init.orthogonal_(self.W_x_rz)
        nn.init.orthogonal_(self.W_h_rz)
        self.b_rz.data.fill_(0)
        nn.init.orthogonal_(self.W_xh)
        nn.init.orthogonal_(self.W_hh)
        self.b_h.data.fill_(0)

    def forward(self, X, hidden, mask = None):
        X_4rz = F.linear(X, self.W_x_rz, self.b_rz)
        X_4h = F.linear(X, self.W_xh, self.b_h) 
        def recurrence(x_4rz, x_4h, hidden, m = None):
            pre_h = hidden
            pre_h = pre_h.view(-1, self.hidden_size)
           
            rz_preact = torch.sigmoid(x_4rz + F.linear(pre_h, self.W_h_rz))
            r, z = rz_preact.chunk(2, 1)
            gh = torch.tanh(x_4h + F.linear(r * pre_h, self.W_hh)) 
            h = (1 - z) * pre_h + z * gh
            
            if m is not None:
                m = m.view(-1, 1)
                h = h * m + pre_h * (1 - m)

            return h

        hiddens = []
        steps = range(X.size(0))
        for i in steps:
            if mask is not None:
                hidden = recurrence(X_4rz[i], X_4h[i], hidden, mask[i])
            else:
                hidden = recurrence(X_4rz[i], X_4h[i], hidden)
            hiddens.append(hidden)

        hiddens = torch.cat(hiddens, 0).view(X.size(0), *hiddens[0].size())
        return hiddens, hidden



