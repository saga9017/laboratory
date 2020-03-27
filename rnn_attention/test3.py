# lstm_attention with torch
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from datetime import datetime
import math
import unicodedata
import re
import random
from random import uniform


view_sentence_len=4
batch_size=2
unknown_number=1

torch.set_default_dtype(torch.double)
torch.autograd.set_detect_anomaly(True)


SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "Null", 3: "UNKNOWN"}
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

    return input_lang, output_lang, pairs


MAX_LENGTH = 10

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


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
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


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print('pair:', random.choice(pairs))

pairs = np.array(pairs)
print(pairs[:, 0][:100])
print(pairs[:, 1][:100])

X_train = []
y_train = []

for sentence in pairs[:, 0]:
    temp = []
    for word in sentence.split():
        if word not in input_lang.word2index:
            temp.append(3)
        else:
            temp.append(input_lang.word2index[word])
    if len(temp) < 10:
        temp.extend([2] * (10 - len(temp)))
    X_train.append(temp)

for sentence in pairs[:, 1]:
    temp = []
    for word in sentence.split():
        if word not in output_lang.word2index:
            temp.append(3)
        else:
            temp.append(output_lang.word2index[word])
    if len(temp) < 10:
        temp.extend([2] * (10 - len(temp)))
    temp.insert(0, 0)
    y_train.append(temp)

print(X_train)
print(y_train)

class LSTM_attention(nn.Module):
    def __init__(self, word_dim1, word_dim2, hidden_dim=256):
        super(LSTM_attention, self).__init__()
        # Assign instance variables
        self.softmax=nn.Softmax(1)
        self.word_dim1 = word_dim1
        self.word_dim2 = word_dim2
        self.hidden_dim = hidden_dim
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        # Randomly initialize the network parameters
        self.U = Parameter(torch.randn(4*hidden_dim, word_dim1))
        self.W = Parameter(torch.randn(4*hidden_dim, hidden_dim))

        self.U_ = Parameter(torch.randn(4*hidden_dim, word_dim2))
        self.W_ = Parameter(torch.randn(4*hidden_dim, hidden_dim))

        self.V = Parameter(torch.randn(word_dim2, 2*hidden_dim))

        self.vU = torch.zeros_like(self.U)
        self.vV = torch.zeros_like(self.V)
        self.vW = torch.zeros_like(self.W)
        self.mU = torch.zeros_like(self.U)
        self.mV = torch.zeros_like(self.V)
        self.mW = torch.zeros_like(self.W)

        self.vU_ = torch.zeros_like(self.U_)
        self.vW_ = torch.zeros_like(self.W_)
        self.mU_ = torch.zeros_like(self.U_)
        self.mW_ = torch.zeros_like(self.W_)

        self.Loss=0

    def forward_propagation(self, x, y):
        # The total number of time steps
        x=torch.tensor(x)
        T1 = len(x[0])
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        h = torch.zeros((T1 + 1, self.hidden_dim, x.shape[0]))
        #h[-1] = torch.zeros((x.shape[0], self.hidden_dim))
        c = torch.zeros((T1 + 1, self.hidden_dim, x.shape[0]))
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

        y = torch.tensor(y)
        T2 = len(y[0])
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = torch.zeros((T2 + 1, self.hidden_dim, x.shape[0]))
        s[-1] = h[-2]
        c_ = torch.zeros((T2 + 1, self.hidden_dim, x.shape[0]))
        c_[-1] = c[-2]
        # The outputs at each time step. Again, we save them for later.
        output = torch.zeros((T2, x.shape[0], self.word_dim2))
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
            e=torch.matmul(h[:-1].transpose(0,2).transpose(1,2), s[t].transpose(0,1).unsqueeze(-1)).view(y.shape[0], -1)
            alpha=self.softmax(e)
            a=torch.matmul(h[:-1].transpose(0,2), alpha.unsqueeze(-1)).view(y.shape[0], -1)
            result=torch.cat((a, s[t].transpose(0,1)), 1)
            output[t]=self.softmax(torch.matmul(result.unsqueeze(1), self.V.transpose(0,1)).view(y.shape[0], -1))

        return output


    def bptt(self, x, y):
        output = self.forward_propagation(x, y)
        y=torch.tensor(y)
        y_ = y[:, 1:]
        y__ = torch.cat((y_, torch.ones((y.shape[0], 1), dtype=torch.int64)), 1)
        loss = -torch.log(torch.gather(output, 2, y__.view(output.shape[0], output.shape[1], 1)))
        loss = torch.sum(loss, dim=1)
        loss = torch.sum(loss, dim=0) / len(x)


        return loss

    def predict(self, x):
        x = torch.tensor(x)
        T1 = len(x[0])
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        h = torch.zeros((T1 + 1, self.hidden_dim, x.shape[0]))
        # h[-1] = torch.zeros((x.shape[0], self.hidden_dim))
        c = torch.zeros((T1 + 1, self.hidden_dim, x.shape[0]))
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

        y = torch.tensor([0] * x.shape[0])
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s_pre = h[-2]
        c__pre = c[-2]
        # The outputs at each time step. Again, we save them for later.
        # For each time step...
        pred=torch.zeros_like(y).view(-1,1)
        step = 0
        while step <= 10:
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            temp_ = self.U_[:, y] + torch.matmul(self.W_.unsqueeze(1), s_pre.clone()).view(-1, y.shape[0])
            i_ = sigmoid(temp_[:H])
            f_ = sigmoid(temp_[H:2 * H])
            o_ = sigmoid(temp_[2 * H:3 * H])
            g_ = torch.tanh(temp_[3 * H:])
            c_ = f_.clone() * ( c__pre.clone()) + i_.clone() * (g_.clone())
            s = o_.clone() * torch.tanh(c_.clone())
            e = torch.matmul(h[:-1].transpose(0, 2).transpose(1, 2), s.transpose(0, 1).unsqueeze(-1)).view(
                y.shape[0], -1)
            alpha = self.softmax(e)
            a = torch.matmul(h[:-1].transpose(0, 2), alpha.unsqueeze(-1)).view(y.shape[0], -1)
            result = torch.cat((a, s.transpose(0, 1)), 1)
            output = self.softmax(torch.matmul(result.unsqueeze(1), self.V.transpose(0, 1)).view(y.shape[0], -1))
            y=torch.argmax(output, dim=1)
            pred=torch.cat((pred, y.view(-1,1)), 1)
            step+=1
        return pred



    # Performs one step of SGD.
    def numpy_sdg_step(self, x, y, learning_rate):
        # Calculate the gradients

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss = self.bptt(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        return loss

    def train_with_batch(self, X_train, y_train, nepoch=100, learning_rate=0.1, evaluate_loss_after=5):
        # We keep track of the losses so we can plot them later
        losses = []
        num_examples_seen = 1
        Loss_len = 0
        max_learning_rate=1
        if len(y_train) % batch_size == 0:
            total_step = int(len(y_train) / batch_size)
        else:
            total_step = int(len(y_train) / batch_size) + 1
        for epoch in range(nepoch):
            # For each training example...
            for i in range(total_step):
                # One SGD step


                if i == total_step - 1:
                    self.Loss += self.numpy_sdg_step(X_train[i * batch_size:],
                                                     y_train[i * batch_size:], learning_rate)
                else:
                    self.Loss += self.numpy_sdg_step(X_train[i * batch_size:(i + 1) * batch_size],
                                                     y_train[i * batch_size:(i + 1) * batch_size], learning_rate)
                Loss_len += 1
                if num_examples_seen == total_step*nepoch:
                    last_loss= self.Loss.item() / Loss_len

                elif num_examples_seen % 10 == 0:
                    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(time, ' ', int(100 * num_examples_seen / (total_step * nepoch)), end='')
                    print('%   완료!!!', end='')
                    print('   loss :', self.Loss.item() / Loss_len)
                    losses.append(self.Loss.item() / Loss_len)
                    if len(losses) > 1:
                        lr_loss = (losses[-2] - losses[-1]) / (losses[-2] + losses[-1])
                        if lr_loss < 0:
                            max_learning_rate=learning_rate
                            learning_rate *= 0.75

                        else:
                            if 1.1*learning_rate>max_learning_rate:
                                pass
                            else:
                                learning_rate += 0.1 * learning_rate

                    print(learning_rate )
                    self.Loss = 0
                    Loss_len = 0
                num_examples_seen += 1

        return last_loss

torch.manual_seed(10)
# Train on a small subset of the data to see what happens
model = LSTM_attention(input_lang.n_words, output_lang.n_words)
#model.gradCheck(X_train[0], y_train[0])
last_loss = model.train_with_batch(X_train, y_train, nepoch=300)


predict1 = model.predict(X_train[:batch_size])
predict2 = model.predict(X_train[batch_size:])


def save(file_name):
    f = open(file_name, 'w')
    f.write('last loss : %s\n' % last_loss)
    f.write('\n')
    for index, k in enumerate([predict1, predict2]):
        for j in range(len(k)):
            f.write('input : %s\n' % pairs[:, 0][j+index*len(k)])
            f.write('result : %s\n' % pairs[:, 1][j+index*len(k)])
            f.write('predict : %s\n' % [output_lang.index2word[i.item()] for i in k[j] if i != 2])
            f.write('\n')

    f.close()
    print("저장 완료!!!")


save('[lstm_attention]view_sentence=4 batch_size=2 epoch=300.txt')