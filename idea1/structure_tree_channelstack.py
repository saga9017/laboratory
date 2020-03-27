#idea1_multichannel2 : residual full, superposition cut-off 5, multichannel3

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
MAX_LENGTH = 30
Batch_size=256
ITERATION=1000
hidden_dim=64
#Dropout=0.1
View_depth1=3
View_depth2=2
View_depth3=3
Layer1=4
Layer2=6
Layer3=4
depth_for_last=2
MAX_OUTPUT=30
Fold=0
#####################################################

class Tree():
    def __init__(self, index):
        self.index=str(index)
        self.child=[]

    def aug(self):
        self.child=[Tree(self.index+'0'), Tree(self.index+'1'), Tree(self.index+'2'), Tree(self.index+'3')]


class suffle_Tree():
    def __init__(self, symbol, operation, depth):
        self.symbol=symbol
        self.operation=operation
        self.depth=depth
        if depth==depth_for_last:
            self.child=None
        else:
            if symbol=='0':
                self.child = [suffle_Tree('+', operation + ' +', depth + 1), suffle_Tree('exp', operation + ' exp', depth + 1)]
            elif symbol=='x1' or symbol=='x2' or symbol=='x3' or symbol=='exp':
                self.child=[suffle_Tree('+', operation+' +', depth+1), suffle_Tree('x', operation+' x', depth+1),
                            suffle_Tree('exp', operation+' exp', depth+1), suffle_Tree('relu', operation+' relu', depth+1)]
            elif symbol=='+' or symbol=='x':
                self.child = [suffle_Tree('x1', operation+' x1', depth+1), suffle_Tree('x2', operation+' x2', depth+1),
                              suffle_Tree('x3', operation+' x3', depth+1)]
            else:
                self.child = [suffle_Tree('+', operation+' +', depth+1), suffle_Tree('x', operation+' x', depth+1),
                              suffle_Tree('exp', operation+' exp', depth+1)]

def last_suffle(Tree):
    temp_trees = Tree.child
    while temp_trees[0].child!=None:
        temp_trees2=[]
        for i in temp_trees:
            temp_trees2.extend(i.child)
        temp_trees=temp_trees2

    symbols=[]
    for i in temp_trees:
        symbols.append(i.operation)


    return symbols



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

    lines = open('content/drive/My Drive/%s-%s.txt' % (lang1, lang2), encoding='utf-8'). \
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



def generate_batch():
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
            temp.extend([0] * (10 - len(temp)))
        X_train.append(temp)

    for sentence in pairs[:, 1]:
        temp = []
        for word in sentence.split():
            if word not in output_lang.word2index:
                temp.append(3)
            else:
                temp.append(output_lang.word2index[word])
        temp.insert(0, len(temp))
        if len(temp) < 10:
            temp.extend([0] * (10 - len(temp)))
        y_train.append(temp)

    batch = []

    if len(y_train) % Batch_size == 0:
        total_step = int(len(y_train) / Batch_size)
    else:
        total_step = int(len(y_train) / Batch_size) + 1

    for i in range(total_step):

        if i == total_step - 1:
            batch.append((X_train[i*Batch_size:], y_train[i*Batch_size:]))
        else:
            batch.append((X_train[i*Batch_size:(i+1)*Batch_size], y_train[i*Batch_size:(i+1)*Batch_size]))

    return batch


class make_combination(nn.Module):
    def __init__(self, view_depth=0):
        super().__init__()
        tree = Tree(0)
        tree.aug()
        depth = 0
        temp_trees = tree.child
        while depth != view_depth - 1:
            depth += 1
            temp = []
            for i in temp_trees:
                i.aug()
                temp.extend(i.child)
            temp_trees = temp

        self.combination = []
        print('tree node 개수 :', len(temp_trees))
        for i in temp_trees:
            if '00' in i.index[1:]:
                pass
            else:
                if '33' in i.index[1:]:
                    pass
                else:
                    self.combination.append(i.index[1:])

        print('combination 수 :', len(self.combination))
        print(self.combination)

    def forward(self):
        return self.combination

class test_sublayer(nn.Module):
    def __init__(self,  out, combination, hidden=hidden_dim, view_depth=0):
        super().__init__()
        self.hidden=hidden
        self.out=out
        self.view_depth=view_depth
        self.combination=combination
        self.term_multi=0.6/(Fold+1)

        self.w0 = Parameter(torch.randn(len(combination), view_depth , 1, hidden))
        self.w1 = Parameter(torch.randn(len(combination), view_depth , hidden, hidden))
        self.w6_1 = nn.Linear(hidden, 4*hidden)
        self.w6_2 = nn.Linear(4*hidden, 1)
        self.w7 = Parameter(torch.randn(len(combination), out))
        self.mask=torch.ones_like(self.w7)
        self.w8 = Parameter(torch.randn(MAX_OUTPUT, out))
        nn.init.xavier_normal_(self.w0)
        nn.init.xavier_normal_(self.w1)
        nn.init.xavier_normal_(self.w6_1.weight)
        nn.init.xavier_normal_(self.w6_2.weight)
        nn.init.xavier_normal_(self.w7)
        nn.init.xavier_normal_(self.w8)

        self.layernorm1 = nn.ModuleList([LayerNorm(hidden)] * view_depth  * len(combination))
        self.layernorm2 = nn.ModuleList([LayerNorm(hidden)] * view_depth  * len(combination))
        self.layernorm3 = nn.ModuleList([LayerNorm(hidden)] * view_depth  * len(combination))

        self.layernorm4 = LayerNorm(len(combination))
        self.layernorm5 = LayerNorm(out)
        self.layernorm6 = LayerNorm(out)



    def forward(self, x):
        result = []
        residual=x
        for index1, i in enumerate(self.combination):

            output = x
            for index2, (a, b) in enumerate(zip(i, self.w1[index1])):
                if a == '0':
                    temp = torch.mm(self.w0[index1][index2], b)
                    output = output + temp
                    output=self.layernorm1[index1 * self.view_depth  + index2](output)
                elif a == '1':
                    output = torch.matmul(output, b)
                    output=self.layernorm2[index1 * self.view_depth  + index2](output)
                elif a == '2':
                    output = torch.exp(output)
                    output=self.layernorm3[index1 * self.view_depth  + index2](output)
                elif a == '3':
                    output = F.relu(output)

            output = self.w6_1(output)
            output = self.w6_2(output)
            result.append(output)

        result = torch.cat(result, dim=2)
        result = self.layernorm4(result)

        #각 fold 마다 0에 가까운 weight만을 남긴다.
        for i in range(Fold):
            if num_examples_seen==int(ITERATION*self.term_multi)*(i+1)+1:
                re_mask=(self.w7 <= 5e-3) & (self.w7 >= -5e-3)
                self.mask = re_mask.float()

        if num_examples_seen> int(ITERATION*self.term_multi):
            if num_examples_seen> ITERATION*0.6:
                new_w = self.w7
            else:
                new_w = self.w7 * self.mask
        else:
            new_w = self.w7

        result = torch.matmul(result, new_w)
        result = self.layernorm5(result)
        if self.hidden==self.out:
            result=result+residual
        result = torch.matmul(self.w8[:, :result.shape[1]], result)
        result = self.layernorm6(result)
        return result, self.mask

def define_layer(view_depth, hidden, layer, word_dim2):
    combination_maker1 = make_combination(view_depth)
    combination = combination_maker1()
    sublayer = [test_sublayer(hidden, combination, hidden, view_depth) for _ in range(layer - 1)]
    sublayer.append(test_sublayer(word_dim2, combination, hidden, view_depth))
    sublayer = nn.ModuleList(sublayer)
    suffle = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layer)])
    for i in suffle:
        nn.init.xavier_normal_(i.weight)

    return layer, sublayer, suffle

class test(nn.Module):
    def __init__(self, word_dim1, word_dim2, hidden=hidden_dim, view_depth1=View_depth1, view_depth2=View_depth2,
                 view_depth3=View_depth3, layer1=Layer1, layer2=Layer2, layer3=Layer3, label_smoothing=0.1):
        super().__init__()

        self.U_e = nn.Embedding(word_dim1, hidden_dim)
        self.U_e.weight.data.uniform_(-0.01, 0.01)


        self.layer1, self.sublayer1, self.suffle1 = define_layer(view_depth1, hidden, layer1, hidden)
        self.layer2, self.sublayer2, self.suffle2 = define_layer(view_depth2, hidden, layer2, hidden)
        self.layer3, self.sublayer3, self.suffle3 = define_layer(view_depth3, hidden, layer3, word_dim2)

        self.symbols = last_suffle(suffle_Tree('0', '0', 0))
        self.w = nn.Linear(len(self.symbols), 1)
        nn.init.xavier_normal_(self.w.weight)

        self.layernorm = LayerNorm(hidden)
        self.layernorm_for_last1= nn.ModuleList([LayerNorm(word_dim2)] * depth_for_last  * len(self.symbols))
        self.layernorm_for_last2 = nn.ModuleList([LayerNorm(word_dim2)] * depth_for_last * len(self.symbols))
        self.layernorm_for_last3 = nn.ModuleList([LayerNorm(word_dim2)] * depth_for_last * len(self.symbols))
        self.softmax=nn.Softmax(-1)

        self.target_prob = Parameter(torch.tensor((1 - label_smoothing) + label_smoothing / word_dim2),
                                     requires_grad=False)
        self.nontarget_prob = Parameter(torch.tensor(label_smoothing / word_dim2), requires_grad=False)

        self.Loss = 0

    def input_embedding(self, x):  # x: (batch, input_len, )
        return self.U_e(x) # (batch, input_len, hidden_dim)

    def OneChannel_forward(self,x, sublayer, suffle, Layer):
        result=x
        total=0
        masks=[]
        for index, (i, j) in enumerate(zip(sublayer, suffle)):
            result = self.layernorm(j(result))
            result, mask = i(result)

            if index!=Layer-1:
                result=result+total
                total=total+result

            masks.append(mask)

        return result, masks

    def cal_symbols(self, symbols, x1, x2, x3):
        outputs = []
        for index1, symbol in enumerate(symbols):
            output = torch.zeros_like(x1)
            stack = None
            for index2, operation in enumerate(symbol.split()):
                if operation == '+':
                    stack = '+'
                elif operation == 'x':
                    stack = 'x'

                elif operation == 'x1':
                    if stack == '+':
                        output = output + x1
                        output = self.layernorm_for_last1[index1 * depth_for_last + index2](output)
                    else:
                        output = output * x1
                        output = self.layernorm_for_last2[index1 * depth_for_last + index2](output)

                elif operation == 'x2':
                    if stack == '+':
                        output = output + x2
                        output = self.layernorm_for_last1[index1 * depth_for_last + index2](output)
                    else:
                        output = output * x2
                        output = self.layernorm_for_last2[index1 * depth_for_last + index2](output)

                elif operation == 'x3':
                    if stack == '+':
                        output = output + x3
                        output = self.layernorm_for_last1[index1 * depth_for_last + index2](output)
                    else:
                        output = output * x3
                        output = self.layernorm_for_last2[index1 * depth_for_last + index2](output)
                elif operation == 'relu':
                    output = F.relu(output)
                else:
                    output = torch.exp(output)
                    output = self.layernorm_for_last3[index1 * depth_for_last + index2](output)

            outputs.append(output.unsqueeze(-1))

        return outputs

    def forward(self, x):

        result = self.input_embedding(x)
        total = 0
        result, masks1=self.OneChannel_forward(result+total, self.sublayer1, self.suffle1, self.layer1)
        total += result
        result, masks2=self.OneChannel_forward(result+total, self.sublayer2, self.suffle2, self.layer2)
        total += result
        result, masks3 = self.OneChannel_forward(result+total, self.sublayer3, self.suffle3, self.layer3)

        result = self.softmax(result)

        return result, masks1, masks2, masks3


    def cal_loss(self, x, y):
        result, masks1, masks2, masks3 = self.forward(x)
        a, b = y.nonzero().t()[0], y.nonzero().t()[1]
        z = result[a, b]
        pos = torch.log(z.gather(1, y[a, b].unsqueeze(-1))).squeeze()
        neg = torch.sum(torch.log(z), dim=1) - pos
        loss = -self.target_prob * pos - self.nontarget_prob * neg
        loss = torch.mean(loss)
        return loss, masks1, masks2, masks3


    def predict(self, x):
        x=torch.tensor(x).to(device)

        result, _, _, _=self.forward(x)
        result=torch.argmax(result, dim=2)

        return result


    def optimizing(self, batch, number_seen, optimizer):
        lrate = math.pow(2048, -0.5) * min(math.pow(number_seen + 1, -0.5), (number_seen + 1) * math.pow(10, -1.5))
        optimizer.param_groups[0]['lr'] = lrate
        optimizer.zero_grad()
        loss, masks1, masks2, masks3=self.cal_loss(torch.tensor(batch[0]).to(device),torch.tensor(batch[1]).to(device))
        loss.backward()
        #0에 가까운 weight는 update 하지 않는다.
        if num_examples_seen > ITERATION*(0.6/(Fold+1)):
            if num_examples_seen > ITERATION * 0.6:
                pass
            else:
                for i, j, a, b, m,n in zip(self.sublayer1, masks1, self.sublayer2, masks2, self.sublayer3, masks3):
                    i.w7.grad = i.w7.grad * j
                    a.w7.grad = a.w7.grad * b
                    m.w7.grad = m.w7.grad * n
        else:
            pass

        optimizer.step()
        #print('loss :', loss.item(), '       lrate :', optimizer.param_groups[0]['lr'])
        return loss

    def train_with_batch(self, batch, evaluate_loss_after=5):
        # We keep track of the losses so we can plot them later
        global num_examples_seen
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

batch= generate_batch()
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
            f.write('input : %s\n' % pairs[:, 0][j+pre_k])
            f.write('result : %s\n' % pairs[:, 1][j+pre_k])
            f.write('predict : %s\n' % [output_lang.index2word[i.item()] for i in k[j] if i != 0][1:1+k[j][0]])
            f.write('\n')
        pre_k+=len(k)
    f.close()
    print("저장 완료!!!")



save('[idea1_multi4]view_sentence='+str(view_sentence_len)+' batch_size='+str(Batch_size)+' iteration='+str(ITERATION)+' layer1='+str(Layer1)+' layer2='+str(Layer2)+' layer3='+str(Layer3)+'.txt')

torch.save(model.state_dict(), 'saved_model')


