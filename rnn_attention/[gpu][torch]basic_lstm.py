# lstm_attention with torch
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from datetime import datetime
import os
import unicodedata
import re
import random
import math
import copy
from random import uniform

os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # GPU number to use

print('gpu :', torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

view_sentence_len=-1       #50000 train, 5000 test
batch_size=2048
unknown_number=1
MAX_LENGTH = 30

torch.autograd.set_detect_anomaly(True)


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
            if unknown_number == 1:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word[self.n_words] = word
                self.n_words += 1
        else:
            if self.word2count[word] >= unknown_number and unknown_number >= 2:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.n_words += 1
            self.word2count[word] += 1


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


def readLangs_train(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines

    lines = open('/content/drive/My Drive/translation/%s-%s_train.txt' % (lang1, lang2), encoding='utf-8'). \
        read().strip().split('\n')


    # 모든 줄을 쌍으로 분리하고 정규화 하십시오
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    pairs = pairs[:view_sentence_len]

    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def readLangs_test(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines

    lines = open('/content/drive/My Drive/translation/%s-%s_test.txt' % (lang1, lang2), encoding='utf-8'). \
        read().strip().split('\n')

    # 모든 줄을 쌍으로 분리하고 정규화 하십시오
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    pairs = pairs[:view_sentence_len]

    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

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
    return len(p[0].split(' ')) < MAX_LENGTH and \
           len(p[1].split(' ')) < MAX_LENGTH \
        #  and p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData_train(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs_train(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

def prepareData_test(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs_test(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

def sigmoid(z):
    return 1 / (1 + torch.exp(-z))


input_lang, output_lang, pairs_train = prepareData_train('eng', 'fra', True)
_, _, pairs_test = prepareData_test('eng', 'fra', True)

pairs_train=np.array(pairs_train)
pairs_test=np.array(pairs_test)


X_train = []
y_train = []
y__train = []

for sentence in pairs_train[:, 0]:
    temp = []
    for word in sentence.split():
        if word not in input_lang.word2index:
            temp.append(3)
        else:
            temp.append(input_lang.word2index[word])
    if len(temp) < MAX_LENGTH:
        temp.extend([0] * (MAX_LENGTH - len(temp)))
    X_train.append(temp)

for sentence in pairs_train[:, 1]:
    temp = []
    temp_ = []
    for word in sentence.split():
        if word not in output_lang.word2index:
            temp.append(3)
        else:
            temp.append(output_lang.word2index[word])
    temp_=copy.deepcopy(temp)
    temp_.append(2)
    if len(temp) < MAX_LENGTH:
        temp.extend([0] * (MAX_LENGTH - len(temp)))
        temp_.extend([0] * (MAX_LENGTH+1 - len(temp_)))
    temp.insert(0, 1)

    y_train.append(temp)
    y__train.append(temp_)

print(X_train[:10])
print(y_train[:10])
print(y__train[:10])


X_test = []
y_test = []

for sentence in pairs_test[:, 0]:
    temp = []
    for word in sentence.split():
        if word not in input_lang.word2index:
            temp.append(3)
        else:
            temp.append(input_lang.word2index[word])
    if len(temp) < MAX_LENGTH:
        temp.extend([0] * (MAX_LENGTH - len(temp)))
    X_test.append(temp)


for sentence in pairs_test[:, 1]:
    temp = []
    for word in sentence.split():
        if word not in output_lang.word2index:
            temp.append(3)
        else:
            temp.append(output_lang.word2index[word])
    temp_=copy.deepcopy(temp)
    if len(temp) < MAX_LENGTH:
        temp.extend([0] * (MAX_LENGTH - len(temp)))
    temp.insert(0, 1)

    y_test.append(temp)

print(X_test[:10])
print(y_test[:10])




class LSTM_attention(nn.Module):
    def __init__(self, word_dim1, word_dim2, hidden_dim=512):
        super(LSTM_attention, self).__init__()
        # Assign instance variables
        self.softmax=nn.Softmax(1)
        self.word_dim1 = word_dim1
        self.word_dim2 = word_dim2
        self.hidden_dim = hidden_dim
        # Randomly initialize the network parameters
        self.U = Parameter(torch.randn(4*hidden_dim, word_dim1).cuda())
        self.W = Parameter(torch.randn(4*hidden_dim, hidden_dim).cuda())
        self.U_ = Parameter(torch.randn(4*hidden_dim, word_dim2).cuda())
        self.W_ = Parameter(torch.randn(4*hidden_dim, hidden_dim).cuda())
        self.V = Parameter(torch.randn(word_dim2, hidden_dim).cuda())
        nn.init.xavier_normal_(self.U)
        nn.init.xavier_normal_(self.W)
        nn.init.xavier_normal_(self.U_)
        nn.init.xavier_normal_(self.W_)
        nn.init.xavier_normal_(self.V)

        self.Loss=0

    def forward_propagation(self, x, y):
        # The total number of time steps
        T1 = len(x[0])
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        h = torch.zeros((T1 + 1, self.hidden_dim, x.shape[0])).cuda()
        #h[-1] = torch.zeros((x.shape[0], self.hidden_dim))
        c = torch.zeros((T1 + 1, self.hidden_dim, x.shape[0])).cuda()
        # The outputs at each time step. Again, we save them for later.
        # For each time step...
        H = self.hidden_dim
        for t in torch.arange(T1):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            temp = self.U[:, x[:, t]] + torch.matmul(self.W.unsqueeze(1), h[t - 1].clone()).view(-1, x.shape[0])
            i = sigmoid(temp[:H])
            f = sigmoid(temp[H:2 * H])
            o = sigmoid(temp[2 * H:3 * H])
            g = torch.tanh(temp[3 * H:])
            c[t] = f.clone() * (c[t - 1].clone()) + i.clone() * (g.clone())
            h[t] = o.clone() * torch.tanh(c[t].clone())

        T2 = len(y[0])
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = torch.zeros((T2 + 1, self.hidden_dim, x.shape[0])).cuda()
        s[-1] = h[-2]
        c_ = torch.zeros((T2 + 1, self.hidden_dim, x.shape[0])).cuda()
        c_[-1] = c[-2]
        # The outputs at each time step. Again, we save them for later.
        output = torch.zeros((T2, x.shape[0], self.word_dim2)).cuda()
        # For each time step...
        for t in torch.arange(T2):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            temp_ = self.U_[:, y[:,t]] + torch.matmul(self.W_.unsqueeze(1), s[t - 1].clone()).view(-1, y.shape[0])
            i_ = sigmoid(temp_[:H])
            f_ = sigmoid(temp_[H:2 * H])
            o_ = sigmoid(temp_[2 * H:3 * H])
            g_ = torch.tanh(temp_[3 * H:])
            c_[t] = f_.clone() * (c_[t - 1].clone()) + i_.clone() * (g_.clone())
            s[t] = o_.clone() * torch.tanh(c_[t].clone())

            output[t]=self.softmax(torch.matmul(s[t].transpose(0,1).clone(), self.V.transpose(0,1)).view(y.shape[0], -1))

        return output



    def bptt(self, x, y, y_):
        output = self.forward_propagation(x, y)
        y_=torch.tensor(y_)
        a, b = y_.nonzero().t()[0], y_.nonzero().t()[1]
        z = output.transpose(0,1)[a, b]
        pos = torch.log(z.gather(1, y_[a, b].unsqueeze(-1))).squeeze()
        loss = -pos
        loss = torch.mean(loss)

        return loss


    def predict(self, x):
        T1 = len(x[0])
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        h = torch.zeros((T1 + 1, self.hidden_dim, x.shape[0])).cuda()
        # h[-1] = torch.zeros((x.shape[0], self.hidden_dim))
        c = torch.zeros((T1 + 1, self.hidden_dim, x.shape[0])).cuda()
        # The outputs at each time step. Again, we save them for later.
        # For each time step...
        H = self.hidden_dim
        for t in torch.arange(T1):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            temp = self.U[:, x[:, t]] + torch.matmul(self.W.unsqueeze(1), h[t - 1].clone()).view(-1, x.shape[0])
            i = sigmoid(temp[:H])
            f = sigmoid(temp[H:2 * H])
            o = sigmoid(temp[2 * H:3 * H])
            g = torch.tanh(temp[3 * H:])
            c[t] = f.clone() * (c[t - 1].clone()) + i.clone() * (g.clone())
            h[t] = o.clone() * torch.tanh(c[t].clone())

        y = torch.tensor([1] * x.shape[0]).cuda()
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s_pre = h[-2]
        c__pre = c[-2]
        # The outputs at each time step. Again, we save them for later.
        # For each time step...
        pred=torch.ones_like(y).view(-1,1).cuda()
        step = 0
        while step < MAX_LENGTH:
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            temp_ = self.U_[:, y] + torch.matmul(self.W_.unsqueeze(1), s_pre.clone()).view(-1, y.shape[0])
            i_ = sigmoid(temp_[:H])
            f_ = sigmoid(temp_[H:2 * H])
            o_ = sigmoid(temp_[2 * H:3 * H])
            g_ = torch.tanh(temp_[3 * H:])
            c_ = f_.clone() * ( c__pre.clone()) + i_.clone() * (g_.clone())
            s = o_.clone() * torch.tanh(c_.clone())
            output = self.softmax(torch.matmul(s.transpose(0,1), self.V.transpose(0, 1)).view(y.shape[0], -1))
            y=torch.argmax(output, dim=1)
            pred=torch.cat((pred, y.view(-1,1)), 1)
            s_pre = s
            c__pre=c_
            step+=1

        return pred



    # Performs one step of SGD.
    def numpy_sdg_step(self, x, y, y_, optimizer):
        # Calculate the gradients
        loss = self.bptt(x, y, y_)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #optimizer.param_groups[0]['lr']=lrate
        return loss


    def train_with_batch(self, X_train, y_train, y__train, nepoch=100, learning_rate=0.01):
        # We keep track of the losses so we can plot them later
        losses = []
        num_examples_seen = 1
        Loss_len = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        if len(y_train) % batch_size == 0:
            total_step = int(len(y_train) / batch_size)
        else:
            total_step = int(len(y_train) / batch_size) + 1
        for epoch in range(nepoch):
            # For each training example...
            for i in range(total_step):
                # One SGD step

                #lrate = math.pow(64, -0.5) * min(math.pow(num_examples_seen, -0.5), num_examples_seen * math.pow(100,  -1.5))  # warm up step : default 4000

                if i == total_step - 1:
                    self.Loss += self.numpy_sdg_step(X_train[i * batch_size:],
                                                     y_train[i * batch_size:], y__train[i * batch_size:], optimizer)
                else:
                    self.Loss += self.numpy_sdg_step(X_train[i * batch_size:(i + 1) * batch_size],
                                                     y_train[i * batch_size:(i + 1) * batch_size],
                                                     y__train[i * batch_size:(i + 1) * batch_size], optimizer)
                Loss_len += 1
                if num_examples_seen == total_step*nepoch:
                    last_loss= self.Loss.item() / Loss_len

                else:
                    if int(total_step * nepoch / 100)==0:
                        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        print(time, ' ', int(100 * num_examples_seen / (total_step * nepoch)), end='')
                        print('%   완료!!!', end='')
                        print('   loss :', self.Loss.item() / Loss_len)
                        losses.append(self.Loss.item() / Loss_len)
                        # print(lrate)
                        self.Loss = 0
                        Loss_len = 0
                    elif num_examples_seen % int(total_step * nepoch / 100) == 0:
                        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        print(time, ' ', int(100 * num_examples_seen / (total_step * nepoch)), end='')
                        print('%   완료!!!', end='')
                        print('   loss :', self.Loss.item() / Loss_len)
                        losses.append(self.Loss.item() / Loss_len)
                        #print(lrate)
                        self.Loss = 0
                        Loss_len = 0
                num_examples_seen += 1

        return last_loss

# Train on a small subset of the data to see what happens
model = LSTM_attention(input_lang.n_words, output_lang.n_words)
model.cuda()
model.to(device)

#model.gradCheck(X_train[0], y_train[0])
last_loss = model.train_with_batch(torch.tensor(X_train).cuda(), torch.tensor(y_train).cuda(), torch.tensor(y__train).cuda(), nepoch=50)


if len(pairs_test) % batch_size==0:
    total_step=int(len(pairs_test)/batch_size)
else:
    total_step=int(len(pairs_test)/batch_size)+1


predicts=[]
for i in range(total_step):
    predicts.append(model.predict(torch.tensor(X_test[i*batch_size:(i+1)*batch_size]).to(device)))



def save(file_name):
    f = open(file_name, 'w')
    f.write('last loss : %s\n' % last_loss)
    f.write('\n')
    #for index, k in enumerate([predict1, predict2, predict3, predict4]):
    for index, k in enumerate(predicts):
        for j in range(len(k)):
            f.write('input : %s\n' % pairs_test[:, 0][j+index*len(k)])
            f.write('result : %s\n' % pairs_test[:, 1][j+index*len(k)])
            f.write('predict : %s\n' % [output_lang.index2word[i.item()] for i in k[j] if i != 0])
            f.write('\n')

    f.close()
    print("저장 완료!!!")


save('[basic_lstm]view_sentence=6669 batch_size=2048 epoch=50.txt')