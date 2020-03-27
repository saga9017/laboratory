# lstm_attention with torch
import numpy
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from datetime import datetime
import sys
import unicodedata
import re
import random
from random import uniform


view_sentence_len=1024

torch.set_default_dtype(torch.double)
torch.autograd.set_detect_anomaly(True)


SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  #  SOS 와 EOS 단어 숫자 포함

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

    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')



    # 모든 줄을 쌍으로 분리하고 정규화 하십시오
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    pairs=pairs[:view_sentence_len]


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


def softmax(a):
    exp_a = torch.exp(a)
    sum_exp_a = torch.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def sigmoid(z):
    return 1 / (1 + torch.exp(-z))


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print('pair:', random.choice(pairs))

pairs=numpy.array(pairs)
print(pairs[:,0][:100])
print(pairs[:,1][:100])

X_train=[]
y_train=[]

for sentence in pairs[:,0]:
    temp=[]
    for word in sentence.split():
        temp.append(input_lang.word2index[word])
    X_train.append(temp)

for sentence in pairs[:,1]:
    temp=[]
    for word in sentence.split():
        temp.append(output_lang.word2index[word])
    temp.insert(0, 0)
    y_train.append(temp)

print(X_train)
print(y_train)

class LSTM_attention(nn.Module):
    def __init__(self, word_dim1, word_dim2, hidden_dim=128):
        super(LSTM_attention, self).__init__()
        # Assign instance variables
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
        T1 = len(x)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        h = torch.zeros((T1 + 1, self.hidden_dim))
        h[-1] = torch.zeros(self.hidden_dim)
        c = torch.zeros((T1 + 1, self.hidden_dim))
        o = torch.zeros((T1, self.hidden_dim))
        i = torch.zeros((T1, self.hidden_dim))
        f = torch.zeros((T1, self.hidden_dim))
        g = torch.zeros((T1, self.hidden_dim))
        # The outputs at each time step. Again, we save them for later.
        # For each time step...
        H = self.hidden_dim
        for t in torch.arange(T1):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            temp = self.U[:, x[t]] + self.W.mm(h[t - 1].clone().view(-1,1)).squeeze()
            i[t] = sigmoid(temp[:H])
            f[t] = sigmoid(temp[H:2 * H])
            o[t] = sigmoid(temp[2 * H:3 * H])
            g[t] = torch.tanh(temp[3 * H:])
            c[t] = f[t].clone() * (c[t - 1].clone()) + i[t].clone() * (g[t].clone())
            h[t] = o[t].clone() * torch.tanh(c[t].clone())


        T2 = len(y)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = torch.zeros((T2 + 1, self.hidden_dim))
        s[-1] = h[-2]
        c_ = torch.zeros((T2 + 1, self.hidden_dim))
        c_[-1] = c[-2]
        o_ = torch.zeros((T2, self.hidden_dim))
        i_ = torch.zeros((T2, self.hidden_dim))
        f_ = torch.zeros((T2, self.hidden_dim))
        g_ = torch.zeros((T2, self.hidden_dim))
        # The outputs at each time step. Again, we save them for later.
        output = torch.zeros((T2, self.word_dim2))
        result=torch.zeros((T2 , self.hidden_dim*2))
        alpha=torch.zeros((T2 , T1))
        # For each time step...
        for t in torch.arange(T2):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            temp_ = self.U_[:, y[t]] + self.W_.mm(s[t - 1].clone().view(-1,1)).squeeze()
            i_[t] = sigmoid(temp_[:H])
            f_[t] = sigmoid(temp_[H:2 * H])
            o_[t] = sigmoid(temp_[2 * H:3 * H])
            g_[t] = torch.tanh(temp_[3 * H:])
            c_[t] = f_[t].clone() * (c_[t - 1].clone()) + i_[t].clone() * (g_[t].clone())
            s[t] = o_[t].clone() * torch.tanh(c_[t].clone())
            e=torch.mm(h[:-1].clone(),s[t].clone().view(-1,1)).squeeze()
            alpha[t]=softmax(e)
            a=torch.mm(alpha[t].clone().view(1,-1), h[:-1].clone()).squeeze()
            result[t]=torch.cat((a, s[t]), 0)
            output[t]=softmax(self.V.mm(result[t].clone().view(-1,1))).squeeze()

        return output


    def bptt(self, x, y):
        T2 = len(y)
        y_=y[1:]
        y_.append(1)
        # Perform forward propagation
        output= self.forward_propagation(x, y)
        # We accumulate the gradients in these variables
        loss = -torch.log(torch.gather(output, 1, torch.tensor(y_).view(output.shape[0], 1)))
        loss=torch.sum(loss)


        return loss

    def predict(self, x):
        T1 = len(x)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        h = torch.zeros((T1 + 1, self.hidden_dim))
        h[-1] = torch.zeros(self.hidden_dim)
        c = torch.zeros((T1 + 1, self.hidden_dim))
        o = torch.zeros((T1, self.hidden_dim))
        i = torch.zeros((T1, self.hidden_dim))
        f = torch.zeros((T1, self.hidden_dim))
        g = torch.zeros((T1, self.hidden_dim))
        # The outputs at each time step. Again, we save them for later.
        # For each time step...
        H = self.hidden_dim
        for t in torch.arange(T1):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            temp = self.U[:, x[t]] + self.W.mm(h[t - 1].clone().view(-1, 1)).squeeze()
            i[t] = sigmoid(temp[:H])
            f[t] = sigmoid(temp[H:2 * H])
            o[t] = sigmoid(temp[2 * H:3 * H])
            g[t] = torch.tanh(temp[3 * H:])
            c[t] = f[t].clone() * (c[t - 1].clone()) + i[t].clone() * (g[t].clone())
            h[t] = o[t].clone() * torch.tanh(c[t].clone())


        s_pre = h[-2]
        c__pre=c[-2]
        z=0
        pred=[]
        att=[]

        # The outputs at each time step. Again, we save them for later.
        output = torch.zeros((self.word_dim2))
        # For each time step...
        step=0
        while z!=1 and step<=10:
            temp_ = self.U_[:, z] + self.W_.mm(s_pre.clone().view(-1, 1)).squeeze()
            i_ = sigmoid(temp_[:H])
            f_ = sigmoid(temp_[H:2 * H])
            o_ = sigmoid(temp_[2 * H:3 * H])
            g_ = torch.tanh(temp_[3 * H:])
            c_ = f_.clone() * (c__pre.clone()) + i_.clone() * (g_.clone())
            s = o_.clone() * torch.tanh(c_.clone())
            e = torch.mm(h[:-1].clone(), s.clone().view(-1, 1)).squeeze()
            alpha = softmax(e)
            a = torch.mm(alpha.clone().view(1, -1), h[:-1].clone()).squeeze()
            result = torch.cat((a, s), 0)
            output = softmax(self.V.mm(result.clone().view(-1, 1))).squeeze()
            z = torch.argmax(output)
            pred.append(z)
            att.append(alpha)
            s_pre = s
            c__pre=c_
            step+=1


        return pred, att



    # Performs one step of SGD.
    def numpy_sdg_step(self, x, y, learning_rate):
        # Calculate the gradients

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss = self.bptt(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def train_with_sgd(self, X_train, y_train, learning_rate=0.0025, nepoch=100):
        # We keep track of the losses so we can plot them later
        losses = []
        num_examples_seen = 1
        Loss_len = 0
        for epoch in range(nepoch):
            # For each training example...
            for i in range(len(y_train)):
                # One SGD step
                self.Loss += self.numpy_sdg_step(X_train[i], y_train[i], learning_rate)
                Loss_len += 1
                if num_examples_seen == len(y_train)*nepoch:
                    last_loss= self.Loss.item() / Loss_len
                elif num_examples_seen % (1024*3) == 0:
                    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(time, ' ', int(100 * num_examples_seen / (len(y_train) * nepoch)), end='')
                    print('%   완료!!!', end='')
                    print('   loss :', self.Loss.item() / Loss_len)
                    self.Loss = 0
                    Loss_len = 0
                num_examples_seen += 1

        return last_loss

# Train on a small subset of the data to see what happens
model = LSTM_attention(input_lang.n_words, output_lang.n_words)
#model.gradCheck(X_train[0], y_train[0])
last_loss = model.train_with_sgd(X_train, y_train, nepoch=300)

def predict(input_number):
    x=pairs[:,0][input_number]
    print('input :', x)
    y=[]

    for word in x.split():
        y.append(input_lang.word2index[word])


    #print('y :', y)
    print('result :', pairs[:,1][input_number])
    #print('predict :', model.predict(y))
    print('predict :', [output_lang.index2word[i.item()] for i in model.predict(y)[0]])
    print()


predict(0)
#predict(5)
#predict(10)
#predict(15)
#predict(20)

predicts=[]
for i in range(view_sentence_len):
    predicts.append(model.predict(X_train[i])[0])


def save(file_name):
    f = open(file_name, 'w')
    f.write('last loss : %s\n' % last_loss)
    f.write('\n')
    for index, k in enumerate(predicts):
        f.write('input : %s\n' % pairs[:, 0][index])
        f.write('result : %s\n' % pairs[:, 1][index])
        print(k)
        f.write('predict : %s\n' % [output_lang.index2word[i.item()] for i in k])
        f.write('\n')

    f.close()
    print("저장 완료!!!")


save('view_sentence=1024 batch_size=1 epoch=300.txt')