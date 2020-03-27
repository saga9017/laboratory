import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from datetime import datetime
import numpy as np
import unicodedata
import re
import random
import math
import os
from preprocessing import matrix_max_len_re_label_dic

os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # GPU number to use

print('gpu :', torch.cuda.is_available())

torch.set_default_dtype(torch.double)
torch.autograd.set_detect_anomaly(True)


batch_size = 128
unknown_number = 1

torch.manual_seed(10)

embedded_matrix, max_len, re_label_dic =matrix_max_len_re_label_dic()



def norm_forward(X, gamma, beta):
    mu = torch.mean(X, dim=2)
    var = torch.var(X, dim=2)
    X_norm = torch.div(X - mu.view(X.shape[0], X.shape[1], 1), torch.sqrt(var.view(X.shape[0], X.shape[1], 1) + 1e-8))
    out = gamma * X_norm + beta
    return out

def norm_forward_in_attention(X, gamma, beta):
    mu = torch.mean(X, dim=3)
    var = torch.var(X, dim=3)
    X_norm = torch.div(X - mu.view(X.shape[0], X.shape[1], X.shape[2], 1), torch.sqrt(var.view(X.shape[0], X.shape[1],X.shape[2],  1) + 1e-8))
    out = (gamma * X_norm.view(8,-1) + beta).view(8, X.shape[1], X.shape[2], X.shape[3])
    return out


def batchnorm_forward(X, gamma, beta):
    mu = torch.mean(X, dim=0)
    var = torch.var(X, dim=0)
    X_norm = torch.div((X - mu), torch.sqrt(var + 1e-8))
    out = gamma * X_norm + beta

    return out

def sigmoid(z):
    return 1 / (1 + torch.exp(-z))



class Encoder(nn.Module):
    def __init__(self, hidden_dim=300, hidden_dim_=512):  # default=512
        # Assign instance variables
        super(Encoder, self).__init__()
        self.hidden = hidden_dim_
        self.softmax = nn.Softmax(3)
        self.WQ_e_L1 = Parameter(torch.randn(hidden_dim_, hidden_dim).cuda())
        self.WK_e_L1 = Parameter(torch.randn(hidden_dim_, hidden_dim).cuda())
        self.WV_e_L1 = Parameter(torch.randn(hidden_dim_, hidden_dim).cuda())
        self.WO_e_L1 = Parameter(torch.randn(hidden_dim_, hidden_dim).cuda())

        self.W1_e = Parameter(torch.randn(hidden_dim, 4 * hidden_dim).cuda())
        self.b1_e = Parameter(torch.randn(4 * hidden_dim).cuda())
        self.W2_e = Parameter(torch.randn(4 * hidden_dim, hidden_dim).cuda())
        self.b2_e = Parameter(torch.randn(hidden_dim).cuda())

        self.gamma_a1 = Parameter(torch.ones((8, 1)).cuda())
        self.beta_a1 = Parameter(torch.zeros((8, 1)).cuda())

        self.gamma1 = Parameter(torch.tensor(1.0).cuda())
        self.beta1 = Parameter(torch.tensor(0.0).cuda())
        self.gamma2 = Parameter(torch.tensor(1.0).cuda())
        self.beta2 = Parameter(torch.tensor(0.0).cuda())


    def multi_head_self_attention(self, x):  # x : (batch_size, input_len, hidden_dim)
        d_k = x.shape[-1]

        q=torch.matmul(x, self.WQ_e_L1.transpose(0,1)).view(x.shape[0], x.shape[1], -1, 8).transpose(3,2).transpose(2,1).transpose(1,0)
        k=torch.matmul(x, self.WK_e_L1.transpose(0,1)).view(x.shape[0], x.shape[1], -1, 8).transpose(3,2).transpose(2,1).transpose(1,0)
        v=torch.matmul(x, self.WV_e_L1.transpose(0,1)).view(x.shape[0], x.shape[1], -1, 8).transpose(3,2).transpose(2,1).transpose(1,0)
        e_pre=torch.matmul(q.clone(), k.clone().transpose(2,3)) / math.sqrt(d_k)
        e = norm_forward_in_attention(e_pre, self.gamma_a1, self.beta_a1)
        alpha=self.softmax(e)
        head1=torch.matmul(alpha.clone(), v.clone())

        a = torch.cat((head1[0], head1[1], head1[2], head1[3], head1[4], head1[5], head1[6], head1[7]), 2)
        result = torch.matmul(a, self.WO_e_L1)

        return result # (input_len, hidden)

    def add_Norm(self, x, y, gamma, beta):  # x : (input_len, hidden)
        e = norm_forward(x + y, gamma, beta)
        return e

    def ffn1(self, x):  # feed forward network   x : (batch_size, input_len, hidden)
        linear1 = torch.matmul(x, self.W1_e) + self.b1_e  # (batch_size, input_len, 4*hidden)
        relu = torch.max(torch.zeros_like(linear1).cuda(), linear1)  # (batch_size, input_len, 4*hidden)
        linear2 = torch.matmul(relu, self.W2_e) + self.b2_e  # (batch_size, input_len, hidden)

        return linear2  # (input_len, 4*hidden)

    def forward(self, x):
        x1=self.multi_head_self_attention(x)
        x2=self.add_Norm(x1, x, self.gamma1, self.beta1)
        x3=self.ffn1(x2)
        x4=self.add_Norm(x3, x2, self.gamma2, self.beta2)

        return x4


class transformer(nn.Module):
    def __init__(self, word_dim2, hidden_dim=300, hidden_dim_=512, label_smoothing=0.1):  # default=512
        # Assign instance variables
        super(transformer, self).__init__()
        self.hidden = hidden_dim_
        self.word_dim2=word_dim2
        self.label_smoothing=label_smoothing
        self.V_e=Parameter(torch.randn(hidden_dim, 1).cuda())
        self.V_d = Parameter(torch.randn(max_len, word_dim2).cuda())

        self.Loss = 0
        self.softmax = nn.Softmax(2)

        self.encoder1=Encoder(hidden_dim, hidden_dim_)
        self.encoder2 = Encoder(hidden_dim, hidden_dim_)
        self.encoder3 = Encoder(hidden_dim, hidden_dim_)
        self.encoder4 = Encoder(hidden_dim, hidden_dim_)
        self.encoder5 = Encoder(hidden_dim, hidden_dim_)
        self.encoder6 = Encoder(hidden_dim, hidden_dim_)

        self.gamma = Parameter(torch.tensor(1.0).cuda())
        self.beta = Parameter(torch.tensor(0.0).cuda())


    def input_embedding(self, x):  # x: (batch, input_len, )
        return embedded_matrix[x]  # (input_len, hidden_dim)


    def Positional_Encoding(self, x):  # x : (batch_size, input_len, hidden_dim) or (batch_size, output_len, hidden_dim)
        table = torch.zeros_like(x[0]).cuda()
        a, b = table.shape
        for pos in range(a):
            for i in range(b):
                if i % 2 == 0:
                    table[pos][i] = math.sin(pos / math.pow(10000, i / b))
                else:
                    table[pos][i] = math.cos(pos / math.pow(10000, (i - 1) / b))
        return x + table


    def forward_propagation(self, x):
        x1 = self.input_embedding(x)  # (input_len, hidden)
        x2 = self.Positional_Encoding(x1)  # (input_len, hidden)
        ########################################################
        x3=self.encoder1.forward(x2+x1)
        x4 = self.encoder2.forward(x3+x2+x1)
        x5 = self.encoder3.forward(x4+x3+x2+x1)
        x6 = self.encoder4.forward(x5+x4+x3+x2+x1)
        x7 = self.encoder5.forward(x6+x5+x4+x3+x2+x1)
        x8 = self.encoder6.forward(x7+x6+x5+x4+x3+x2+x1)
        x9 = torch.matmul(x8, self.V_e)
        ########################################################
        x10 = self.softmax(norm_forward(x9.transpose(1,2).matmul(self.V_d), self.gamma, self.beta)).view(x9.shape[0], -1)
        return x10

    def bptt(self, x, y):  # (batch_size, out_len)
        x10 = self.forward_propagation(x)
        loss=torch.sum(-torch.log(torch.gather(x10, 1, y.view(-1,1))))/x.shape[0]
        return loss

    def predict(self, x):
        x1 = self.input_embedding(x)  # (input_len, hidden)
        x2 = self.Positional_Encoding(x1)  # (input_len, hidden)
        ########################################################
        x3 = self.encoder1.forward(x2 + x1)
        x4 = self.encoder2.forward(x3 + x2 + x1)
        x5 = self.encoder3.forward(x4 + x3 + x2 + x1)
        x6 = self.encoder4.forward(x5 + x4 + x3 + x2 + x1)
        x7 = self.encoder5.forward(x6 + x5 + x4 + x3 + x2 + x1)
        x8 = self.encoder6.forward(x7 + x6 + x5 + x4 + x3 + x2 + x1)
        x9 = torch.matmul(x8, self.V_e)
        ########################################################
        x10 = self.softmax(norm_forward(x9.transpose(1, 2).matmul(self.V_d), self.gamma, self.beta)).view(x9.shape[0],
                                                                                                          -1)
        output=torch.tensor([re_label_dic[i.item()] for i in torch.argmax(x10, 1)]).cuda()
        return output

    # Performs one step of SGD.
    def numpy_sdg_step(self, x, y, learning_rate):
        # Calculate the gradients
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss = self.bptt(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def epoch_eval(self, x_dev, y_dev):
        score=torch.tensor(0.0)

        if len(y_dev) % batch_size == 0:
            total_step = int(len(y_dev) / batch_size)
        else:
            total_step = int(len(y_dev) / batch_size) + 1

        for i in range(total_step):
            if i == total_step - 1:
                flags=(self.predict(x_dev[i * batch_size:])==y_dev[i * batch_size:])
                score+=torch.sum(flags)
            else:
                flags = (self.predict(x_dev[i * batch_size:(i + 1) * batch_size]) == y_dev[i * batch_size:(i + 1) * batch_size])
                score += torch.sum(flags)

        return score/len(y_dev)


    def train_with_batch(self, X_train, y_train,  X_dev, y_dev, nepoch=100, evaluate_loss_after=5):
        # We keep track of the losses so we can plot them later
        losses = []
        num_examples_seen = 1
        Loss_len = 0
        if len(y_train) % batch_size == 0:
            total_step = int(len(y_train) / batch_size)
        else:
            total_step = int(len(y_train) / batch_size) + 1
        for epoch in range(nepoch):
            # For each training example...
            for i in range(total_step):
                # One SGD step
                lrate = math.pow(self.hidden, -0.5) * min(math.pow(num_examples_seen, -0.5),
                                                          num_examples_seen * math.pow(400, -1.5))

                if i == total_step - 1:
                    self.Loss += self.numpy_sdg_step(X_train[i * batch_size:],
                                                     y_train[i * batch_size:], lrate )
                else:
                    self.Loss += self.numpy_sdg_step(X_train[i * batch_size:(i + 1) * batch_size],
                                                     y_train[i * batch_size:(i + 1) * batch_size], lrate )
                Loss_len += 1
                if num_examples_seen == total_step*nepoch:
                    last_loss= self.Loss.item() / Loss_len

                elif num_examples_seen % 10 == 0:
                    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(time, ' ', int(100 * num_examples_seen / (total_step * nepoch)), end='')
                    print('%   완료!!!', end='')
                    print('    lr :', lrate , '   loss :', self.Loss.item() / Loss_len)
                    self.Loss = 0
                    Loss_len = 0
                num_examples_seen += 1

            print('dev accuracy :', self.epoch_eval(X_dev, y_dev).item())
        return last_loss