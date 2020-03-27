#transformer : encoder, decoder 6, use residual

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional  as F
from datetime import datetime
import numpy as np
import unicodedata
import re
import random
import math
import os
import time


os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # GPU number to use

print('gpu :', torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_default_dtype(torch.float)

print('cudnn :', torch.backends.cudnn.enabled == True)
torch.backends.cudnn.benchmark=True
#####################################################
view_sentence_len = 10
unknown_number = 1
MAX_LENGTH = 175
MAX_TOKEN=1024
ITERATION=100*1000
hidden_dim=128
Dropout=0.1
#####################################################

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "Null", 1: "SOS", 2: "EOS",  3: "UNKNOWN"}
        self.n_words = 4  # SOS 와 EOS 와 Null 단어 숫자 포함

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1

        else:
            self.word2count[word] += 1

    def make_dic(self):
        word2count_copy=self.word2count.copy()
        self.word2count={"UNKNOWN":0}
        self.word2index = {}
        self.n_words=4
        self.index2word = {0: "Null", 1: "SOS", 2: "EOS", 3: "UNKNOWN"}
        for i in word2count_copy.keys():
            if word2count_copy[i]>=unknown_number:
                self.word2index[i] = self.n_words
                self.index2word[self.n_words] = i
                self.word2count[i]=word2count_copy[i]
                self.n_words += 1
            else:
                self.word2count["UNKNOWN"] += word2count_copy[i]

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# 소문자, 다듬기, 그리고 문자가 아닌 문자 제거


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines

    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8'). \
        read().strip().split('\n')

    # 모든 줄을 쌍으로 분리하고 정규화 하십시오
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    pairs = pairs[:view_sentence_len]

    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
    print(pairs)

    return input_lang, output_lang, pairs


eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    print(p[0])
    return len(p[0].split(' ')) < MAX_LENGTH and \
           len(p[1].split(' ')) < MAX_LENGTH \
        #  and p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    input_lang.make_dic()
    output_lang.make_dic()
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print('pair:', random.choice(pairs))

pairs = np.array(pairs)
print(pairs[:, 0][:10])
print(pairs[:, 1][:10])

def Positional_Encoding(MAX_LENGTH, hidden_dim):  # x : (batch_size, input_len, hidden_dim) or (batch_size, output_len, hidden_dim)
    table = torch.zeros((MAX_LENGTH, hidden_dim))
    a, b = table.shape
    for pos in range(a):
        for i in range(b):
            if i % 2 == 0:
                table[pos][i] = math.sin(pos / math.pow(10000, i / b))
            else:
                table[pos][i] = math.cos(pos / math.pow(10000, (i - 1) / b))
    return Parameter(table, requires_grad=False)

table={}
for i in range(MAX_LENGTH+1):
    table[i+1]=Positional_Encoding(i+1, hidden_dim)

mask={}
for i in range(MAX_LENGTH+1):
    mask[i+1] = torch.triu(torch.ones((i+1, i+1)), diagonal=1).byte()


def generate_batch(MAX_TOKEN):
    X_train = []  # 만들어진 순서 + word index로 이루어짐
    y_train = []
    y__train = []
    for sentence in pairs[:, 0]:
        temp = [len(X_train)]
        for word in sentence.split():
            if word not in input_lang.word2index:
                temp.append(3)
            else:
                temp.append(input_lang.word2index[word])
        X_train.append(temp)

    for sentence in pairs[:, 1]:
        temp = []
        temp_ = []
        for word in sentence.split():
            if word not in output_lang.word2index:
                temp.append(3)
                temp_.append(3)
            else:
                temp.append(output_lang.word2index[word])
                temp_.append(output_lang.word2index[word])
        temp.insert(0, 1)
        temp_.append(2)

        y_train.append(temp)
        y__train.append(temp_)

    temp_X = []
    temp_y = []
    temp_y_ = []
    batch = []

    key = list(reversed(sorted(X_train, key=len)))
    key_dic={}

    k = len(key[0][1:])

    max_num_sen = int(MAX_TOKEN / k)
    num_sen = 0
    max_len_y = 0
    for i in key:
        key_dic[len(key_dic)]=i[0]
        if len(i[1:]) <= k:
            temp_X.append(i[1:] + [0] * (k - len(i[1:])))
        else:
            print('error')
            temp_X.append(i[1:])

        temp_y.append(y_train[i[0]])
        temp_y_.append(y__train[i[0]])
        if len(y_train[i[0]]) > max_len_y:
            max_len_y = len(y_train[i[0]])
        num_sen += 1
        if num_sen == max_num_sen:
            k = len(key[num_sen][1:])
            max_num_sen = int(MAX_TOKEN / k)
            num_sen = 0
            for index, i in enumerate(temp_y):
                if len(i) <= max_len_y:
                    temp_y[index] = temp_y[index] + [0] * (max_len_y - len(i))
                    temp_y_[index] = temp_y_[index] + [0] * (max_len_y - len(i))

            batch.append((temp_X, len(temp_X[0]), temp_y, len(temp_y[0]),temp_y_, len(temp_y[0])))
            temp_X = []
            temp_y = []
            temp_y_ = []
            max_len_y = 0

    for index, i in enumerate(temp_y):
        if len(i) <= max_len_y:
            temp_y[index] = temp_y[index] + [0] * (max_len_y - len(i))
            temp_y_[index] = temp_y_[index] + [0] * (max_len_y - len(i))


    batch.append((temp_X, len(temp_X[0]), temp_y, len(temp_y[0]),temp_y_, len(temp_y[0])))

    return batch, key_dic

def get_pad_mask(x):
    z=torch.zeros(x.shape[0], x.shape[1], hidden_dim)
    result=x.nonzero()

    return result
#################################################### Module ############################################################

class Norm(nn.Module):
    def __init__(self):
        super(Norm, self).__init__()
        self.gamma=Parameter(torch.tensor(1.0))
        self.beta = Parameter(torch.tensor(0.0))
    def forward(self, X):
        mu = torch.mean(X, dim=2)
        var = torch.var(X, dim=2)
        X_norm = torch.div(X - mu.view(X.shape[0], X.shape[1], 1), torch.sqrt(var.view(X.shape[0], X.shape[1], 1) + 1e-8))
        out = self.gamma * X_norm + self.beta
        return out

class Multi_head_attention(nn.Module):
    def __init__(self, hidden_dim=300, hidden_dim_=512, dropout=Dropout):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.softmax = nn.Softmax(-1)
        self.layerNorm_add_Norm = Norm()

        self.w_qs = nn.Linear(hidden_dim, hidden_dim_)
        self.w_ks = nn.Linear(hidden_dim, hidden_dim_)
        self.w_vs = nn.Linear(hidden_dim, hidden_dim_)
        self.w_os = nn.Linear(hidden_dim_, hidden_dim)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (hidden_dim)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (hidden_dim)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (hidden_dim)))
        nn.init.xavier_normal_(self.w_os.weight)

    def forward(self, en, de, mask, dropout=False):  # x : (input_len, hidden_dim)
        d_k = de.shape[-1]
        len_d0, len_d1=de.shape[0], de.shape[1]
        len_e0, len_e1=en.shape[0], en.shape[1]


        q = self.w_qs(de).view(len_d0, len_d1, -1, 8).permute(3, 0, 1, 2)
        k = self.w_ks(en).view(len_e0, len_e1, -1, 8).permute(3, 0, 2, 1)
        v = self.w_vs(en).view(len_e0, len_e1, -1, 8).permute(3, 0, 1, 2)

        e = torch.matmul(q, k) / math.sqrt(d_k)
        masked_e = e.masked_fill(mask, -1e10)
        alpha = self.softmax(masked_e)  # (output_len, input_len)
        if dropout==True:
            alpha = self.dropout(alpha)
        head3 = torch.matmul(alpha, v)

        a = torch.cat((head3[0], head3[1], head3[2], head3[3], head3[4], head3[5], head3[6], head3[7]), 2)

        result = self.w_os(a)
        result=self.layerNorm_add_Norm(result+de)
        return result  # (output_len, hidden)

class FFN(nn.Module):  # feed forward network   x : (batch_size, input_len, hidden)
    def __init__(self, hidden_dim=300, dropout=Dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layerNorm_add_Norm = Norm()

        self.fc1 = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.fc2 = nn.Linear(4*hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)


    def forward(self, x, dropout=False):
        output = self.fc1(x) # (batch_size, input_len, 4*hidden)
        if dropout==True:
            output = self.dropout(F.relu(output))  # (batch_size, input_len, 4*hidden)
        else:
            output = F.relu(output)
        output = self.fc2(output)  # (batch_size, input_len, hidden
        output=self.layerNorm_add_Norm(output+x)
        return output

##################################################### Sub layer ########################################################

class Encoder_layer(nn.Module):
    def __init__(self, hidden_dim=300, hidden_dim_=512, dropout=Dropout):  # default=512
        # Assign instance variables
        super().__init__()
        self.multi_head_self_attention=Multi_head_attention(hidden_dim, hidden_dim_, dropout)
        self.ffn=FFN(hidden_dim, dropout)


        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)


    def forward(self, x, mask, non_pad_mask, dropout=False):
        if dropout==True:
            output=self.dropout_1(self.multi_head_self_attention(x, x, mask, dropout=True))
            output=output.masked_fill(non_pad_mask==0, 0)
            output=self.dropout_2(self.ffn(output, dropout=True))
            output=output.masked_fill(non_pad_mask==0, 0)
        else:
            output = self.multi_head_self_attention(x, x, mask)
            output = output.masked_fill(non_pad_mask == 0, 0)
            output = self.ffn(output)
            output = output.masked_fill(non_pad_mask == 0, 0)
        return output

class Decoder_layer(nn.Module):
    def __init__(self, hidden_dim=300, hidden_dim_=512, dropout=Dropout):  # default=512
        # Assign instance variables
        super().__init__()

        self.masked_multi_head_self_attention = Multi_head_attention(hidden_dim, hidden_dim_, dropout)
        self.multi_head_attention=Multi_head_attention(hidden_dim, hidden_dim_, dropout)
        self.ffn=FFN(hidden_dim, dropout)



        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)


    def forward(self, y, en, mask1, mask2, non_pad_mask_y=1, dropout=False):
        if dropout==True:
            output = self.dropout_1(self.masked_multi_head_self_attention(y, y, mask1, dropout=True))
            output = output*non_pad_mask_y
            output = self.dropout_2(self.multi_head_attention(en, output, mask2, dropout=True))
            output = output*non_pad_mask_y
            output = self.dropout_3(self.ffn(output, dropout=True))
            output = output*non_pad_mask_y
        else:
            output = self.masked_multi_head_self_attention(y, y, mask1)
            output = output * non_pad_mask_y
            output = self.multi_head_attention(en, output, mask2)
            output = output * non_pad_mask_y
            output = self.ffn(output)
            output = output * non_pad_mask_y
        return output

#################################################### Layer ##############################################################
class Encoder(nn.Module):
    def __init__(self, word_dim1, hidden_dim=hidden_dim, hidden_dim_=hidden_dim):  # default=512
        # Assign instance variables
        super().__init__()
        self.hidden = hidden_dim
        self.U_e = nn.Embedding(word_dim1, hidden_dim)

        self.encoder1=Encoder_layer(hidden_dim, hidden_dim_)
        self.encoder2 = Encoder_layer(hidden_dim, hidden_dim_)
        self.encoder3 = Encoder_layer(hidden_dim, hidden_dim_)
        self.encoder4 = Encoder_layer(hidden_dim, hidden_dim_)
        self.encoder5 = Encoder_layer(hidden_dim, hidden_dim_)
        self.encoder6 = Encoder_layer(hidden_dim, hidden_dim_)

    def input_embedding(self, x):  # x: (batch, input_len, )
        mask_e=Parameter((x==0).unsqueeze(1).repeat(1,x.shape[1], 1), requires_grad=False)
        mask_d = Parameter((x == 0), requires_grad=False)
        return self.U_e(x), mask_e, mask_d  # (input_len, hidden_dim)

    def forward(self, x, table_x, dropout):
        non_pad_mask = (x != 0).unsqueeze(-1).repeat(1, 1, self.hidden).float()
        x1, mask_e, mask_d = self.input_embedding(x)  # (input_len, hidden)
        x1=x1.masked_fill(non_pad_mask==0, 0)
        x2 = (x1 + table[table_x].to(device)).masked_fill(non_pad_mask==0, 0)  # (input_len, hidden)
        #x2 = x1 + table_x
        ########################################################
        x3 = self.encoder1(x2 + x1, mask_e, non_pad_mask, dropout=dropout)
        x4 = self.encoder2(x3 + x2 + x1, mask_e, non_pad_mask, dropout=dropout)
        x5 = self.encoder3(x4 + x3 + x2 + x1, mask_e, non_pad_mask, dropout=dropout)
        x6 = self.encoder4(x5 + x4 + x3 + x2 + x1, mask_e, non_pad_mask, dropout=dropout)
        x7 = self.encoder5(x6 + x5 + x4 + x3 + x2 + x1, mask_e, non_pad_mask, dropout=dropout)
        x8 = self.encoder6(x7 + x6 + x5 + x4 + x3 + x2 + x1, mask_e, non_pad_mask, dropout=dropout)

        return x8, mask_d


class Decoder(nn.Module):
    def __init__(self, word_dim2, hidden_dim=hidden_dim, hidden_dim_=hidden_dim):  # default=512
        # Assign instance variables
        super().__init__()
        self.hidden = hidden_dim
        self.word_dim2 = word_dim2
        self.softmax = nn.Softmax(-1)

        self.U_d = nn.Embedding(word_dim2, hidden_dim)
        self.V_d = nn.Linear(hidden_dim, word_dim2, bias=False)
        nn.init.xavier_normal_(self.V_d.weight)


        self.decoder1 = Decoder_layer(hidden_dim, hidden_dim_)
        self.decoder2 = Decoder_layer(hidden_dim, hidden_dim_)
        self.decoder3 = Decoder_layer(hidden_dim, hidden_dim_)
        self.decoder4 = Decoder_layer(hidden_dim, hidden_dim_)
        self.decoder5 = Decoder_layer(hidden_dim, hidden_dim_)
        self.decoder6 = Decoder_layer(hidden_dim, hidden_dim_)


    def output_embedding(self, y):  # x: (batch_size, output_len, )
        mask = Parameter((y == 0), requires_grad=False)
        return self.U_d(y), mask  # (output_len, hidden_dim)

    def forward(self, x8, mask_d, y, table_y, mask_, dropout):
        non_pad_mask_y = (y != 0).unsqueeze(-1).repeat(1, 1, self.hidden).float()
        y1, mask_dd = self.output_embedding(y)  # (output_len,hidden)
        y1=y1.masked_fill(non_pad_mask_y==0, 0)
        mask_dd = mask_dd.unsqueeze(1).repeat(1, y.shape[1], 1) | mask[mask_].to(device)
        y2 = (y1+table[table_y].to(device)).masked_fill(non_pad_mask_y==0, 0) # (output_len,hidden)
        #y2 = y1 + table_y
        mask_d = mask_d.unsqueeze(1).repeat(1, y.shape[1], 1)
        ########################################################
        y3 = self.decoder1(y2 + y1, x8, mask_dd, mask_d, non_pad_mask_y, dropout=dropout)
        y4 = self.decoder2(y3 + y2 + y1, x8, mask_dd, mask_d, non_pad_mask_y, dropout=dropout)
        y5 = self.decoder3(y4 + y3 + y2 + y1, x8, mask_dd, mask_d, non_pad_mask_y, dropout=dropout)
        y6 = self.decoder4(y5 + y4 + y3 + y2 + y1, x8, mask_dd, mask_d, non_pad_mask_y, dropout=dropout)
        y7 = self.decoder5(y6 + y5 + y4 + y3 + y2 + y1, x8, mask_dd, mask_d, non_pad_mask_y, dropout=dropout)
        y8 = self.decoder6(y7 + y6 + y5 + y4 + y3 + y2 + y1, x8, mask_dd, mask_d, non_pad_mask_y, dropout=dropout)
        #######################################################
        y9 = self.softmax(self.V_d(y8))

        return y9



class transformer(nn.Module):
    def __init__(self, word_dim1, word_dim2, hidden_dim=hidden_dim, hidden_dim_=hidden_dim, label_smoothing=0.1):  # default=512
        # Assign instance variables
        super(transformer, self).__init__()
        self.hidden = hidden_dim

        self.Loss = 0

        self.encoder=Encoder(word_dim1, hidden_dim, hidden_dim_)
        self.decoder=Decoder(word_dim2, hidden_dim, hidden_dim_)

        self.target_prob = Parameter(torch.tensor((1 - label_smoothing) + label_smoothing / word_dim2), requires_grad=False)
        self.nontarget_prob = Parameter(torch.tensor(label_smoothing / word_dim2), requires_grad=False)


    def forward_propagation(self, x, table_x, y, table_y, mask):
        x, mask_d=self.encoder(x, table_x, dropout=True)
        y=self.decoder(x, mask_d, y, table_y, mask, dropout=True)
        return y


    def bptt(self, x, table_x, y, table_y, y_, mask):  # (batch_size, out_len)
        y9 = self.forward_propagation(x, table_x, y, table_y, mask)

        a,b=y_.nonzero().t()[0], y_.nonzero().t()[1]
        z=y9[a,b]
        pos=torch.log(z.gather(1, y_[a,b].unsqueeze(-1))).squeeze()
        neg=torch.sum(torch.log(z), dim=1)-pos
        loss = -self.target_prob * pos - self.nontarget_prob * neg
        loss=torch.mean(loss)
        return loss


    def predict(self, x, table_):
        x=torch.tensor(x).to(device)
        ###############################################################################################################
        x, mask_d=self.encoder(x, table_, dropout=False)
        ###############################################################################################################
        output = torch.tensor([[1]] * x.shape[0]).to(device)
        step = 0
        while step < MAX_LENGTH+1:
            y=self.decoder(x, mask_d, output, step+1, step+1, dropout=False)
            output = torch.cat((output, torch.argmax(y[:,-1], dim=1).unsqueeze(-1)), 1)
            step += 1
        return output

    def beam_search(self, x, table_, beam_number=2):
        x=torch.tensor(x).to(device)
        table_=table_.to(device)
        ###############################################################################################################
        x, mask_d=self.encoder(x, table_, dropout=False)
        ###############################################################################################################
        output = torch.tensor([[1]] * x.shape[0]).to(device)
        y = self.decoder(x, mask_d, output, table[1].to(device), mask[1].to(device), dropout=False)
        candidate=torch.topk(y[:,-1], beam_number, dim=1)
        can_output=[]
        can_prob = candidate[0]
        for i in range(beam_number):
            print(candidate[1][i].shape)
            can_output.append(torch.cat((output, candidate[1][i]), 1))
        step = 1
        while step < MAX_LENGTH+1:
            can_prob_twice=torch.zeros(beam_number**2)
            can_output_twice=[]
            for j in range(beam_number):
                for i in range(beam_number):
                    y=self.decoder(x, mask_d, can_output[i], table[step+1].to(device), mask[step+1].to(device), dropout=False)
                    candidate = torch.topk(y[:, -1], beam_number, dim=1)
                    can_prob_twice[i+(j*i)]=can_prob[j]*candidate[0][i]
                    can_output_twice.append(torch.cat((output[j], candidate[1][i]), 1))
            candidate=torch.topk(can_prob_twice, beam_number)
            can_prob = candidate[0]
            for i in range(beam_number):
                print(candidate[1][i].shape)
                can_output=can_output_twice[candidate[1][i].item()]
            output = torch.cat((output, torch.argmax(y[:, -1], dim=1).unsqueeze(-1)), 1)
            step += 1
        return output


    # Performs one step of SGD.
    def numpy_sdg_step(self, batch, optimizer):
        # Calculate the gradients

        optimizer.zero_grad()
        loss = self.bptt(torch.tensor(batch[0]).to(device), batch[1], torch.tensor(batch[2]).to(device),
                         batch[3], torch.tensor(batch[4]).to(device), batch[5])
        loss.backward()
        optimizer.step()

        return loss

    def train_with_batch(self, batch, evaluate_loss_after=5):
        # We keep track of the losses so we can plot them later
        num_examples_seen = 1
        Loss_len = 0
        last_loss=0
        nepoch=int(ITERATION/len(batch))
        print('epoch :', nepoch)
        optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=0.00002)
        for epoch in range(nepoch):
            # For each training example...
            for i in range(len(batch)):
                # One SGD step
                self.Loss += self.numpy_sdg_step(batch[i], optimizer).item()

                Loss_len += 1
                if num_examples_seen == len(batch)*nepoch:
                    last_loss= self.Loss / Loss_len
                else:
                    if int(len(batch) * nepoch/100)==0:
                        time_ = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        print(time_, ' ', int(100 * num_examples_seen / (len(batch) * nepoch)), end='')
                        print('%   완료!!!', end='')
                        print('   loss :', self.Loss / Loss_len)
                        self.Loss = 0
                        Loss_len = 0
                    else:
                        if num_examples_seen %  int(len(batch) * nepoch/100)== 0:
                            time_ = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            print(time_, ' ', int(100 * num_examples_seen / (len(batch) * nepoch)), end='')
                            print('%   완료!!!', end='')
                            print('   loss :', self.Loss / Loss_len)
                            self.Loss = 0
                            Loss_len = 0
                num_examples_seen += 1
        return last_loss

torch.manual_seed(10)
# Train on a small subset of the data to see what happens
model = transformer(input_lang.n_words, output_lang.n_words).to(device)

"""""
for parameter in model.parameters():
    print(parameter)
"""""

batch, key_dic= generate_batch(MAX_TOKEN)
print('preprocess done!!!')
last_loss = model.train_with_batch(batch)




predicts=[]
for i in range(len(batch)):
    predicts.append(model.predict(batch[i][0], batch[i][1]))

    if i%10==0:
        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(time, ' ', 100*i/len(batch), '%   저장!!!')




def save(file_name):
    f = open(file_name, 'w')
    f.write('last loss : %s\n' % last_loss)
    f.write('\n')
    pre_k=0
    for index, k in enumerate(predicts):
        for j in range(len(k)):
            f.write('input : %s\n' % pairs[:, 0][key_dic[j+pre_k]])
            f.write('result : %s\n' % pairs[:, 1][key_dic[j+pre_k]])
            f.write('predict : %s\n' % [output_lang.index2word[i.item()] for i in k[j] if i != 0])
            f.write('\n')
        pre_k+=len(k)
    f.close()
    print("저장 완료!!!")



save('[final]view_sentence='+str(view_sentence_len)+' batch_size='+str(MAX_TOKEN)+' iteration='+str(ITERATION)+'.txt')

torch.save(model.state_dict(), 'saved_model')


