#pylint: skip-file
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from lstm import *
#from lstm_peepholes import *
#from lstm_peepholes_bad import *
from gru import *

class RNN(nn.Module):
    def __init__(self, out_size, hidden_size, batch_size, dim_w, dict_size, local_rnn = False, cell = "gru", num_layers = 1):
        super(RNN, self).__init__()
        self.in_size = dim_w
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.cell = cell.lower()
        self.batch_size = batch_size
        self.dict_size = dict_size
        self.dim_w = dim_w
        self.num_layers = num_layers
        self.local_rnn = local_rnn

        self.raw_emb = nn.Embedding(self.dict_size, self.dim_w, 0)
        if self.cell == "gru":
            if self.local_rnn: 
                self.rnn_model = GRU(self.in_size, self.hidden_size)
            else:
                self.rnn_model = nn.GRU(self.in_size, self.hidden_size)
        elif self.cell == "lstm":
            if self.local_rnn:
                self.rnn_model = LSTM(self.in_size, self.hidden_size)
            else:
                self.rnn_model = nn.LSTM(self.in_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.out_size)
    
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.raw_emb.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, len_x, hidden=None, mask=None):
        emb_x = self.raw_emb(x)
        if self.local_rnn:
            hs, hn = self.rnn_model(emb_x, hidden, mask)
        else:
            self.rnn_model.flatten_parameters()
            
            emb_x = torch.nn.utils.rnn.pack_padded_sequence(emb_x, len_x)
            hs, hn = self.rnn_model(emb_x, hidden)
            hs, _ = torch.nn.utils.rnn.pad_packed_sequence(hs)
        
        output = self.linear(hs) 
        return output, hn
    
    def init_hidden(self, batch_size):
        if self.cell == "lstm":
            return (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)))
        return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))

