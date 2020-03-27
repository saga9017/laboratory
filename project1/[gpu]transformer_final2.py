#transformer : encoder, decoder 6, use residual

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional  as F
from datetime import datetime
import numpy as np
import math
import os
import pickle


os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # GPU number to use

print('gpu :', torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_default_dtype(torch.double)
torch.autograd.set_detect_anomaly(True)
#####################################################
view_sentence_len = -1
unknown_number = 1
MAX_LENGTH = 30
MAX_TOKEN=3000
nepoch=10
#####################################################

m = dict()
v = dict()

with open('/content/drive/My Drive/w2v_lower.pkl', 'rb') as fin:
    embedded = pickle.load(fin)


embedded_matrix=torch.from_numpy(np.concatenate((np.zeros((1,300)), embedded[0]), 0)).cuda()
w2i=embedded[1]

y_train=[]
X_train=[]

max_len=1
label_dic={}
re_label_dic={}
label_num=0
with open('/content/drive/My Drive/cnn_data/odp.wsdm.train.all', 'r') as f:
    while True:
        line = f.readline().replace(',', '').split()
        if not line: break
        if int(line[0]) not in label_dic:
            label_dic[int(line[0])]=label_num
            re_label_dic[label_num]=int(line[0])
            label_num+=1

        y_train.append(label_dic[int(line[0])])
        temp=[len(X_train)]
        for index, i in enumerate(line[1:]):
            if index+1>max_len:
                max_len=index+1

            if i not in w2i.keys():
                temp.append(w2i['unknown'])
            else:
                temp.append(w2i[i])
        X_train.append(temp)

print(X_train[:10])
print(y_train[:10])

y_dev=[]
X_dev=[]
with open('/content/drive/My Drive/cnn_data/odp.wsdm.dev.all', 'r') as f:
    while True:
        line = f.readline().replace(',', '').split()
        if not line: break

        y_dev.append(int(line[0]))
        temp=[len(X_dev)]
        for index, i in enumerate(line[1:]):
            if index + 1 > max_len:
                continue

            if i not in w2i.keys():
                temp.append(w2i['unknown'])
            else:
                temp.append(w2i[i])
        X_dev.append(temp)

print(X_dev[:10])
print(y_dev[:10])

def generate_batch(X_train, y_train, MAX_TOKEN):

    temp_X = []
    temp_y = []
    batch = []

    key = list(reversed(sorted(X_train, key=len)))
    key_dic={}

    k = len(key[0][1:])

    max_num_sen = int(MAX_TOKEN / k)
    num_sen = 0
    for i in key:
        key_dic[len(key_dic)]=i[0]
        if len(i[1:]) <= k:
            temp_X.append(i[1:] + [0] * (k - len(i[1:])))
        else:
            print('error')
            temp_X.append(i[1:])

        temp_y.append(y_train[i[0]])

        num_sen += 1
        if num_sen == max_num_sen:
            k = len(key[num_sen][1:])
            max_num_sen = int(MAX_TOKEN / k)
            num_sen = 0
            batch.append((temp_X, temp_y))
            temp_X = []
            temp_y = []



    batch.append((temp_X, temp_y))
    return batch, key_dic



def Positional_Encoding(MAX_LENGTH, hidden_dim):  # x : (batch_size, input_len, hidden_dim) or (batch_size, output_len, hidden_dim)
    table = torch.zeros((MAX_LENGTH, hidden_dim)).cuda()
    a, b = table.shape
    for pos in range(a):
        for i in range(b):
            if i % 2 == 0:
                table[pos][i] = math.sin(pos / math.pow(10000, i / b))
            else:
                table[pos][i] = math.cos(pos / math.pow(10000, (i - 1) / b))
    return Parameter(table, requires_grad=False)


class Norm(nn.Module):
    def __init__(self):
        super(Norm, self).__init__()
        self.gamma=Parameter(torch.tensor(1.0).cuda())
        self.beta = Parameter(torch.tensor(0.0).cuda())
    def forward(self, X):
        mu = torch.mean(X, dim=-1, keepdim=True)
        var = torch.var(X, dim=-1, keepdim=True)
        X_norm = torch.div(X - mu, torch.sqrt(var + 1e-8))
        out = self.gamma * X_norm + self.beta
        return out

class Multi_head_attention(nn.Module):
    def __init__(self, hidden_dim=300, hidden_dim_=512, dropout=0.1):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.softmax = nn.Softmax(3)
        self.WQ = Parameter(torch.randn(hidden_dim, hidden_dim_).cuda())
        self.WK = Parameter(torch.randn(hidden_dim, hidden_dim_).cuda())
        self.WV = Parameter(torch.randn(hidden_dim, hidden_dim_).cuda())
        self.WO = Parameter(torch.randn(hidden_dim_, hidden_dim).cuda())


    def forward(self, en, de, mask):  # x : (input_len, hidden_dim)
        d_k = de.shape[-1]
        len_d0, len_d1=de.shape[0], de.shape[1]
        len_e0, len_e1=en.shape[0], en.shape[1]

        q = torch.matmul(de, self.WQ).view(len_d0, len_d1 , -1, 8).permute(3, 0, 1, 2)
        k = torch.matmul(en ,self.WK).view(len_e0, len_e1, -1, 8).permute(3, 0, 2, 1)
        v = torch.matmul(en, self.WV).view(len_e0, len_e1, -1, 8).permute(3, 0, 1, 2)

        e = torch.matmul(q, k) / math.sqrt(d_k)
        masked_e = e.masked_fill(mask, -1e9)
        alpha = self.softmax(masked_e)  # (output_len, input_len)
        alpha = self.dropout(alpha)
        head3 = torch.matmul(alpha, v)

        a = torch.cat((head3[0], head3[1], head3[2], head3[3], head3[4], head3[5], head3[6], head3[7]), 2)
        result = torch.matmul(a, self.WO)
        return result  # (output_len, hidden)

class FFN(nn.Module):  # feed forward network   x : (batch_size, input_len, hidden)
    def __init__(self, hidden_dim=300, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.W1 = Parameter(torch.randn(hidden_dim, 4 * hidden_dim).cuda())
        self.b1=Parameter(torch.randn(4 * hidden_dim).cuda())
        self.W2 = Parameter(torch.randn(4*hidden_dim, hidden_dim).cuda())
        self.b2 = Parameter(torch.randn(hidden_dim).cuda())


    def forward(self, x):
        linear1 = torch.matmul(x, self.W1)+self.b1  # (batch_size, input_len, 4*hidden)
        relu = self.dropout(F.relu(linear1))  # (batch_size, input_len, 4*hidden)
        linear2 = torch.matmul(relu, self.W2)+self.b2  # (batch_size, input_len, hidden)
        return linear2


class Encoder(nn.Module):
    def __init__(self, hidden_dim=300, hidden_dim_=512, dropout=0.1):  # default=512
        # Assign instance variables
        super().__init__()
        self.multi_head_self_attention=Multi_head_attention(hidden_dim, hidden_dim_, dropout)
        self.ffn=FFN(hidden_dim, dropout)

        self.layerNorm_add_Norm1=Norm()
        self.layerNorm_add_Norm2 = Norm()

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)


    def forward(self, x, mask):
        x2=self.dropout_1(self.multi_head_self_attention(x, x, mask))
        x=self.layerNorm_add_Norm1(x2+x)
        x2 = self.dropout_2(self.ffn(x))
        x=self.layerNorm_add_Norm2(x2+x)
        return x



class transformer(nn.Module):
    def __init__(self, word_dim2, hidden_dim=300, hidden_dim_=64, label_smoothing=0.1):  # default=512
        # Assign instance variables
        super(transformer, self).__init__()
        self.hidden = hidden_dim
        self.word_dim2 = word_dim2
        self.V_d = Parameter(torch.randn(hidden_dim, word_dim2).cuda())
        self.Loss = 0
        self.softmax = nn.Softmax(1)

        self.encoder1=Encoder(hidden_dim, hidden_dim_)
        self.encoder2 = Encoder(hidden_dim, hidden_dim_)
        self.encoder3 = Encoder(hidden_dim, hidden_dim_)
        self.encoder4 = Encoder(hidden_dim, hidden_dim_)
        self.encoder5 = Encoder(hidden_dim, hidden_dim_)
        self.encoder6 = Encoder(hidden_dim, hidden_dim_)
        self.layerNorm = Norm()



    def input_embedding(self, x):  # x: (batch, input_len, )
        mask_e=Parameter((x==0).unsqueeze(1).repeat(1,x.shape[1], 1).cuda(), requires_grad=False)
        return embedded_matrix[x], mask_e  # (input_len, hidden_dim)


    def forward_propagation(self, x):
        x1, mask_e= self.input_embedding(x)  # (input_len, hidden)
        x2 = x1 + Positional_Encoding(x.shape[1], self.hidden)  # (input_len, hidden)
        ########################################################
        x3 = self.encoder1(x2 + x1, mask_e)
        x4 = self.encoder2(x3 + x2 + x1, mask_e)
        x5 = self.encoder3(x4 + x3 + x2 + x1, mask_e)
        x6 = self.encoder4(x5 + x4 + x3 + x2 + x1, mask_e)
        x7 = self.encoder5(x6 + x5 + x4 + x3 + x2 + x1, mask_e)
        x8 = self.encoder6(x7 + x6 + x5 + x4 + x3 + x2 + x1, mask_e)
        ########################################################
        x9 = self.softmax(self.layerNorm(torch.matmul(torch.sum(x8, dim=1), self.V_d))).view(x8.shape[0], -1)
        return x9

    def bptt(self, x, y):  # (batch_size, out_len)
        x9 = self.forward_propagation(x)

        loss = torch.mean(-torch.log(torch.gather(x9, 1, y.view(-1, 1))))
        return loss


    def predict(self, x):
        self.eval()
        x1, mask_e = self.input_embedding(x)  # (input_len, hidden)
        x2 = x1 + Positional_Encoding(x.shape[1], self.hidden)  # (input_len, hidden)
        ########################################################
        x3 = self.encoder1(x2 + x1, mask_e)
        x4 = self.encoder2(x3 + x2 + x1, mask_e)
        x5 = self.encoder3(x4 + x3 + x2 + x1, mask_e)
        x6 = self.encoder4(x5 + x4 + x3 + x2 + x1, mask_e)
        x7 = self.encoder5(x6 + x5 + x4 + x3 + x2 + x1, mask_e)
        x8 = self.encoder6(x7 + x6 + x5 + x4 + x3 + x2 + x1, mask_e)
        ########################################################
        x9 = self.softmax(self.layerNorm(torch.matmul(torch.sum(x8, dim=1), self.V_d))).view(x8.shape[0], -1)
        output = torch.tensor([re_label_dic[i.item()] for i in torch.argmax(x9, 1)])

        return output
    def epoch_eval(self, batch_dev):
        score=torch.tensor(0.0)
        num_flags=0
        for i in range(len(batch_dev)):
            flags = (self.predict(torch.tensor(batch_dev[i][0]).cuda()) == torch.tensor(batch_dev[i][1]).cuda())
            num_flags+=len(flags)
            score += torch.sum(flags)

        return 100*score/num_flags

    # Performs one step of SGD.
    def numpy_sdg_step(self, batch, learning_rate):
        # Calculate the gradients
        optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=learning_rate)
        optimizer.zero_grad()
        loss = self.bptt(torch.tensor(batch[0]).cuda(), torch.tensor(batch[1]).cuda())
        loss.backward()
        optimizer.step()

        return loss

    def train_with_batch(self, batch, batch_dev, nepoch, evaluate_loss_after=5):
        # We keep track of the losses so we can plot them later
        num_examples_seen = 1
        Loss_len = 0


        for epoch in range(nepoch):
            # For each training example...
            for i in range(len(batch)):
                # One SGD step
                lrate = math.pow(self.hidden, -0.5) * min(math.pow(num_examples_seen, -0.5),
                                                          num_examples_seen * math.pow(10, -1.5))  # warm up step : default 4000

                self.Loss += self.numpy_sdg_step(batch[i], lrate).item()

                Loss_len += 1
                if num_examples_seen == len(batch)*nepoch:
                    last_loss= self.Loss / Loss_len
                elif num_examples_seen % int(len(batch)*nepoch/100) == 0:
                    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(time, ' ', int(100 * num_examples_seen / (len(batch) * nepoch)), end='')
                    print('%   완료!!!', end='')
                    print('   loss :', self.Loss / Loss_len)
                    self.Loss = 0
                    Loss_len = 0

                num_examples_seen += 1
            print('dev accuracy :', self.epoch_eval(batch_dev).item())
        return last_loss

torch.manual_seed(10)
# Train on a small subset of the data to see what happens
model = transformer(max(y_train)+1)
model.to(device)

"""""
for parameter in model.parameters():
    print(parameter)
"""""
batch, _= generate_batch(X_train, y_train, MAX_TOKEN)
batch_dev,_ =generate_batch(X_dev, y_dev, MAX_TOKEN)
last_loss = model.train_with_batch(batch, batch_dev, nepoch=nepoch)




predicts=[]
for i in range(len(batch)):
    predicts.append(model.predict(batch[i][0]))

    if i%10==0:
        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(time, ' ', 100*i/len(batch), '%   저장!!!')



torch.save(model.state_dict(), 'saved_model')


