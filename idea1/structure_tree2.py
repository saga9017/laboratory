#transformer : encoder, decoder 6, use residual

import torch
import torch.nn as nn
from torch.nn import LayerNorm
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
view_sentence_len = 1024
unknown_number = 1
MAX_LENGTH = 175
MAX_TOKEN=512
ITERATION=1000
hidden_dim=512
#Dropout=0.1
view_depth=5
MAX_OUTPUT=50
MAX_INPUT=50
#####################################################

class Tree():
    def __init__(self, index):
        self.index=str(index)
        self.child=[]

    def aug(self):
        self.child=[Tree(self.index+'0'), Tree(self.index+'1'), Tree(self.index+'2'), Tree(self.index+'3')]

tree=Tree(0)
tree.aug()
depth=0
temp_trees=tree.child
while depth!=view_depth:
    depth+=1
    temp=[]
    for i in temp_trees:
        i.aug()
        temp.extend(i.child)
    temp_trees=temp

combination=[]
print('tree node 개수 :',len(temp_trees))
for i in temp_trees:
    if '00' in i.index[1:]:
        pass
    else:
        if '33' in i.index[1:]:
            pass
        else:
            combination.append(i.index[1:])


print('combination 수 :', len(combination))
print(combination)


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
        temp.insert(0, len(temp))
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





class test(nn.Module):
    def __init__(self, word_dim1, word_dim2, hidden=hidden_dim, label_smoothing=0.1):
        super().__init__()

        self.U_e = nn.Embedding(word_dim1, hidden_dim)

        self.w0 = Parameter(torch.randn(view_depth+1, 1, hidden))
        self.w1=Parameter(torch.randn(view_depth+1, hidden,hidden))
        self.w6=nn.Linear(hidden,1)
        self.w7=Parameter(torch.randn(len(combination), word_dim2))
        self.w8=Parameter(torch.randn(MAX_OUTPUT, word_dim2))
        nn.init.xavier_normal_(self.w0)
        nn.init.xavier_normal_(self.w1)
        nn.init.xavier_normal_(self.w6.weight)
        nn.init.xavier_normal_(self.w7)
        nn.init.xavier_normal_(self.w8)

        self.layernorm1 = nn.ModuleList([LayerNorm(hidden)]*(view_depth+1)*len(combination))
        self.layernorm2 = nn.ModuleList([LayerNorm(hidden)]*(view_depth+1)*len(combination))
        self.layernorm3 = nn.ModuleList([LayerNorm(hidden)] * (view_depth + 1) * len(combination))

        self.layernorm4 = LayerNorm(len(combination))
        self.layernorm5 = LayerNorm(word_dim2)
        self.layernorm6 = LayerNorm(word_dim2)
        self.softmax=nn.Softmax(-1)

        self.target_prob = Parameter(torch.tensor((1 - label_smoothing) + label_smoothing / word_dim2),
                                     requires_grad=False)
        self.nontarget_prob = Parameter(torch.tensor(label_smoothing / word_dim2), requires_grad=False)

        self.Loss = 0

    def input_embedding(self, x):  # x: (batch, input_len, )
        return self.U_e(x) # (batch, input_len, hidden_dim)

    def forward(self,x):
        result = []

        for index1, i in enumerate(combination):

            output = self.input_embedding(x)
            for index2, (a,b) in enumerate(zip(i, self.w1)):
                if a == '0':
                    temp=torch.mm(self.w0[index2], b)
                    output=output+temp
                    self.layernorm1[index1 * (view_depth + 1) + index2](output)
                elif a == '1':
                    output=torch.matmul(output, b)
                    output=self.layernorm2[index1*(view_depth+1)+index2](output)
                elif a == '2':
                    output=torch.exp(output)
                    output=self.layernorm3[index1*(view_depth+1)+index2](output)
                elif a == '3':
                    output=F.relu(output)


            output=self.w6(output)
            result.append(output)

        result=torch.cat(result, dim=2)
        result=self.layernorm4(result)
        mask7=1-(self.w7<=1e-7).float()
        new_w7=self.w7*mask7
        result=torch.matmul(result, new_w7)
        result = self.layernorm5(result)
        result=torch.matmul(self.w8[:, :result.shape[1]], result)
        result = self.layernorm6(result)
        result=self.softmax(result)

        return result, mask7

    def cal_loss(self, x, y):
        result, mask = self.forward(x)

        a, b = y.nonzero().t()[0], y.nonzero().t()[1]
        z = result[a, b]
        pos = torch.log(z.gather(1, y[a, b].unsqueeze(-1))).squeeze()
        neg = torch.sum(torch.log(z), dim=1) - pos
        loss = -self.target_prob * pos - self.nontarget_prob * neg
        loss = torch.mean(loss)
        return loss, mask


    def predict(self, x):
        x=torch.tensor(x).to(device)

        result, _=self.forward(x)
        result=torch.argmax(result, dim=2)

        return result


    def optimizing(self, batch, number_seen, optimizer):
        lrate = math.pow(1024, -0.5) * min(math.pow(number_seen + 1, -0.5), (number_seen + 1) * math.pow(10, -1.5))
        optimizer.param_groups[0]['lr'] = lrate
        optimizer.zero_grad()
        loss, mask=self.cal_loss(torch.tensor(batch[0]).to(device),torch.tensor(batch[2]).to(device))
        loss.backward()
        model.w7.grad = model.w7.grad * mask
        optimizer.step()
        #print('loss :', loss.item(), '       lrate :', optimizer.param_groups[0]['lr'])

        return loss

    def train_with_batch(self, batch, evaluate_loss_after=5):
        # We keep track of the losses so we can plot them later
        num_examples_seen = 1
        Loss_len = 0
        last_loss=0
        nepoch=int(ITERATION/len(batch))
        print('epoch :', nepoch)
        optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=0.0)
        for epoch in range(nepoch):
            # For each training example...
            for i in range(len(batch)):
                # One SGD step
                self.Loss += self.optimizing(batch[i], num_examples_seen, optimizer).item()

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
model = test(input_lang.n_words, output_lang.n_words).to(device)

"""""
for parameter in model.parameters():
    print(parameter)
"""""

batch, key_dic= generate_batch(MAX_TOKEN)
print('preprocess done!!!')
last_loss = model.train_with_batch(batch)




predicts=[]
for i in range(len(batch)):
    predicts.append(model.predict(batch[i][0]))

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
            f.write('predict : %s\n' % [output_lang.index2word[i.item()] for i in k[j] if i != 0][:k[j][0]])
            f.write('\n')
        pre_k+=len(k)
    f.close()
    print("저장 완료!!!")



save('[final]view_sentence='+str(view_sentence_len)+' batch_size='+str(MAX_TOKEN)+' iteration='+str(ITERATION)+'.txt')

torch.save(model.state_dict(), 'saved_model')


