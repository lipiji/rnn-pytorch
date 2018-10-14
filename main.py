#pylint: skip-file
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import sys
import numpy as np
from rnn import *
import data
  

use_gpu = True

local_rnn = False # true: lstm.py; false: nn.LSTM

lr = 1
drop_rate = 0.
batch_size = 128
hidden_size = 500
emb_size = 300
cell = "lstm"

use_cuda = use_gpu and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

dtype = torch.FloatTensor

xy_list, i2w, w2i, batch_list = data.char_sequence("/data/shakespeare.txt", batch_size)
dict_size = len(w2i)
num_batches = len(batch_list)
print "dict_size=", dict_size, "#batchs=", num_batches, "#batch_size", batch_size

print "compiling..."
model = RNN(dict_size, hidden_size, batch_size, emb_size, dict_size, local_rnn, cell)
optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)

weight = torch.ones(dict_size)
weight[0] = 0.0
if use_cuda:
    model.cuda()
    weight = weight.cuda()
criterion = nn.CrossEntropyLoss(weight)
#def train(x, y):


print "training..."
hidden = model.init_hidden(batch_size)
if use_cuda:
    if model.cell == "gru":
        hidden = hidden.cuda()
    else:
        hidden = (hidden[0].cuda(), hidden[1].cuda())

start = time.time()
for i in xrange(100):
    error = 0.0    
    in_start = time.time()
    for idx_batch in xrange(num_batches):
        train_idx = batch_list[idx_batch]
        batch_raw = [xy_list[xy_idx] for xy_idx in train_idx]
        if len(batch_raw) != batch_size:
            continue
        len_x = 0
        for seq in batch_raw:
            if len(seq) > len_x:
                len_x = len(seq)
        len_x -= 1
        local_batch_size = len(batch_raw)
        batch = data.get_data(batch_raw, len_x, w2i, i2w)
        sorted_x_idx = np.argsort(batch.x_len)[::-1]
        sorted_x_len = np.array(batch.x_len)[sorted_x_idx]
        sorted_x = batch.x[:, sorted_x_idx]
        sorted_x_mask = batch.x_mask[:, sorted_x_idx]

        sorted_y_len = np.array(batch.y_len)[sorted_x_idx]
        sorted_y = batch.y[:, sorted_x_idx]

        model.zero_grad()
        out, hn = model(torch.LongTensor(sorted_x).to(device), torch.LongTensor(sorted_x_len).to(device), \
                        hidden, torch.FloatTensor(sorted_x_mask).to(device))
        cost = criterion(out.view(-1, len(w2i)), torch.LongTensor(sorted_y).contiguous().view(-1).cuda())
        cost.backward()
        optimizer.step()

        cost = cost.item()
        error += cost
        #print idx_batch, num_batches, cost
        
    in_time = time.time() - in_start

    error /= num_batches;

    print "Iter = " + str(i) + ", Loss = " + str(error) + ", Time = " + str(in_time)

print "Finished. Time = " + str(time.time() - start)

print "save model..."
if not os.path.exists("./model/"):
    os.makedirs("./model/")
filepath = "./model/char_rnn.model"
torch.save({"model_state_dict" : model.state_dict(),
            "optimizer_state_dict" : optimizer.state_dict()},
            filepath)
    

def generate(model, prime_str='t', predict_len=100, temperature=0.8, cuda=use_cuda):
    x = np.zeros((1, 1), dtype = np.int64)
    x_mask = np.zeros((1, 1))
    x[0, 0] = w2i[prime_str]
    x_mask[0, 0] = 1
    len_x = [1]
    x = torch.LongTensor(x)
    x_mask = torch.FloatTensor(x_mask)
    len_x = torch.LongTensor(len_x)
    if cuda:
        len_x = len_x.cuda()
        x_mask = x_mask.cuda()
    predicted = prime_str 
    
    hidden = model.init_hidden(1)

    for p in range(predict_len):
        if cuda:
            x = x.cuda()
            if model.cell == "gru":
                hidden = hidden.cuda()
            else:
                hidden = (hidden[0].cuda(), hidden[1].cuda())

        output, hidden = model(x, len_x, hidden, x_mask)
       
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1).item()

        predicted_char = i2w[top_i]
        predicted += predicted_char
        x[0, 0] = top_i
        #x = torch.LongTensor(x)
    return predicted

print "Testing..."
print "load model..."
checkpoint = torch.load(filepath)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

print generate(model, prime_str='t', predict_len=2000)


