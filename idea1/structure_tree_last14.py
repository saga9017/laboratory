# idea1_multichannel2 : residual full, superposition cut-off 5, multichannel3, integration

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
import copy
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # GPU number to use

print('gpu :', torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_default_dtype(torch.float)
torch.autograd.set_detect_anomaly(True)

print('cudnn :', torch.backends.cudnn.enabled == True)
torch.backends.cudnn.benchmark = True
#####################################################
view_sentence_len = 1024
unknown_number = 1
MAX_LENGTH = 30
Batch_size = 512
ITERATION = 1000
hidden_dim = 256
# Dropout=0.1
View_depth1 = 1
Layer1 = 12
MAX_OUTPUT = 30
Fold = 0
Integration_object = 3  # 홀수
Integration_number = 4  # 짝수


#####################################################

class Tree():
    def __init__(self, index):
        self.index = str(index)
        self.child = []

    def aug(self):
        self.child = [Tree(self.index + '0'), Tree(self.index + '1'), Tree(self.index + '2'), Tree(self.index + '3')]


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "Null", 1: "SOS", 2: "EOS", 3: "UNKNOWN"}
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
        word2count_copy = self.word2count.copy()
        self.word2count = {"UNKNOWN": 0}
        self.word2index = {}
        self.n_words = 4
        self.index2word = {0: "Null", 1: "SOS", 2: "EOS", 3: "UNKNOWN"}
        for i in word2count_copy.keys():
            if word2count_copy[i] >= unknown_number:
                self.word2index[i] = self.n_words
                self.index2word[self.n_words] = i
                self.word2count[i] = word2count_copy[i]
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


def generate_batch():
    X_train = []
    y_train = []
    y__train = []

    for sentence in pairs[:, 0]:
        temp = []
        for word in sentence.split():
            if word not in input_lang.word2index:
                temp.append(3)
            else:
                temp.append(input_lang.word2index[word])
        if len(temp) <= MAX_LENGTH:
            temp.extend([0] * (MAX_LENGTH - len(temp)))
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
        if len(temp) <= MAX_LENGTH:
            temp.extend([0] * (MAX_LENGTH - len(temp)))
            temp_.extend([0] * (MAX_LENGTH - len(temp_)))
        y_train.append(temp)
        y__train.append(temp_)

    batch = []

    if len(y_train) % Batch_size == 0:
        total_step = int(len(y_train) / Batch_size)
    else:
        total_step = int(len(y_train) / Batch_size) + 1

    for i in range(total_step):

        if i == total_step - 1:
            batch.append((X_train[i * Batch_size:], y_train[i * Batch_size:], y__train[i * Batch_size:]))
        else:
            batch.append((X_train[i * Batch_size:(i + 1) * Batch_size], y_train[i * Batch_size:(i + 1) * Batch_size],
                          y__train[i * Batch_size:(i + 1) * Batch_size]))

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
    def __init__(self, out, combination, hidden=hidden_dim, view_depth=0):
        super().__init__()
        self.hidden = hidden
        self.out = out
        self.view_depth = view_depth
        self.combination = combination
        self.term_multi = 0.6 / (Fold + 1)

        self.w0 = Parameter(torch.randn(1, hidden))
        self.w1 = Parameter(torch.randn(hidden, hidden))
        self.w6_2 = nn.Linear(hidden, out)

        self.w8 = Parameter(torch.randn(MAX_OUTPUT, MAX_OUTPUT))
        nn.init.xavier_normal_(self.w0)
        nn.init.xavier_normal_(self.w1)
        nn.init.xavier_normal_(self.w6_2.weight)
        nn.init.xavier_normal_(self.w8)

        self.layernorm3 = LayerNorm(hidden)

        self.layernorm6 = LayerNorm(out)

    def cal_output(self, output, result):
        if self.hidden != self.out:
            output = self.w6_2(output)
        result.append(output.unsqueeze(-1))

    def forward(self, x):
        result = []
        total_out = x

        temp = torch.mm(self.w0, self.w1)
        output = x + temp
        self.cal_output(output, result)

        output = torch.matmul(x.clone(), self.w1)
        self.cal_output(output, result)

        output = self.layernorm3(x.clone())
        output = torch.exp(output)
        self.cal_output(output, result)

        output = F.relu(x.clone())
        self.cal_output(output, result)

        result = torch.cat(result, dim=-1)
        result = torch.max(result, dim=-1)[0]

        if self.hidden == self.out:
            result += total_out
            total_out += result
        else:
            pass

        if result.shape[1] == MAX_OUTPUT:
            if self.hidden == self.out:
                result = torch.matmul(self.w8[:, :result.shape[1]], result) + total_out
            else:
                result = torch.matmul(self.w8[:, :result.shape[1]], result) + result
        else:
            result = torch.matmul(self.w8[:, :result.shape[1]], result)

        result = self.layernorm6(result)

        return result


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


class Decoder(nn.Module):
    def __init__(self, word_dim2):
        super(Decoder, self).__init__()
        self.U_d = nn.Embedding(word_dim2, hidden_dim)
        self.U_d.weight.data.uniform_(-0.01, 0.01)
        self.w = nn.Linear(hidden_dim, word_dim2)
        self.w2_1 = nn.Linear(word_dim2*2, hidden_dim)
        self.w2_2 = nn.Linear(hidden_dim, word_dim2)
        self.layernorm = LayerNorm(word_dim2)
        self.softmax = nn.Softmax(-1)

    def forward(self, e_out, y):
        result = self.U_d(y)
        result = self.w(result)

        result = torch.cat([e_out, result], dim=-1)
        result=self.w2_1(result)
        result=self.w2_2(result)

        result = self.softmax(self.layernorm(result))
        return result


class test(nn.Module):
    def __init__(self, word_dim1, word_dim2, hidden=hidden_dim, view_depth1=View_depth1,
                 layer1=Layer1, label_smoothing=0.1):
        super().__init__()

        self.U_e = nn.Embedding(word_dim1, hidden_dim)
        self.U_e.weight.data.uniform_(-0.01, 0.01)
        self.decoder = Decoder(word_dim2)

        self.layer1, self.sublayer1, self.suffle1 = define_layer(view_depth1, hidden, layer1, word_dim2)

        self.w1 = nn.Linear(word_dim2, hidden)
        self.w2 = nn.Linear(hidden, word_dim2)
        nn.init.xavier_normal_(self.w1.weight)
        nn.init.xavier_normal_(self.w2.weight)

        self.target_prob = (1 - label_smoothing) + label_smoothing / word_dim2
        self.nontarget_prob = label_smoothing / word_dim2

        self.Loss = 0

    def input_embedding(self, x):  # x: (batch, input_len, )a
        return self.U_e(x)  # (batch, input_len, hidden_dim)

    def OneChannel_forward(self, x, sublayer, suffle, Layer):
        result = self.input_embedding(x)
        residual = 0

        for index, (i, j) in enumerate(zip(sublayer[:-1], suffle[:-1])):
            result = j(result)
            result = i(result)

            if index % 2 == 0:
                result += residual
                residual = result

        result = suffle[-1](result)
        result = sublayer[-1](result)
        result = self.w1(result)
        result = self.w2(result)

        return result

    def encoding(self, x):
        result = self.OneChannel_forward(x, self.sublayer1, self.suffle1, self.layer1)

        return result

    def forward(self, x, y):

        result = self.encoding(x)
        #########################################
        result = self.decoder(result, y)

        return result

    def cal_loss(self, x, y, y_):
        result = self.forward(x, y)
        a, b = y_.nonzero().t()
        z = result[a, b]
        pos = torch.log(z.gather(1, y_[a, b].unsqueeze(-1))).squeeze()
        neg = torch.sum(torch.log(z), dim=1) - pos
        loss = -self.target_prob * pos - self.nontarget_prob * neg
        loss = torch.mean(loss)
        return loss

    def predict(self, x):
        x = torch.tensor(x).to(device)

        result = self.encoding(x)
        output = torch.ones(x.shape[0]).unsqueeze(-1).long().to(device)
        for i in range(MAX_OUTPUT):
            output2 = self.decoder(result[:, :i + 1], output)
            output = torch.cat([output, torch.argmax(output2, dim=2)[:, -1].unsqueeze(-1)], dim=1)

        return output

    def optimizing(self, batch, number_seen, optimizer):
        lrate = math.pow(512, -0.5) * min(math.pow(number_seen + 1, -0.5), (number_seen + 1) * math.pow(10, -1.5))
        optimizer.param_groups[0]['lr'] = lrate
        optimizer.zero_grad()
        loss = self.cal_loss(torch.tensor(batch[0]).to(device), torch.tensor(batch[1]).to(device),
                             torch.tensor(batch[2]).to(device))
        loss.backward()

        optimizer.step()
        # print('loss :', loss.item(), '       lrate :', optimizer.param_groups[0]['lr'])
        return loss

    def train_with_batch(self, batch, optimizer, iteration=0, Num_examples_seen=1):
        # We keep track of the losses so we can plot them later
        global num_examples_seen
        num_examples_seen = Num_examples_seen
        Loss_len = 0
        last_loss = 0
        if iteration % len(batch) == 0:
            nepoch = int(iteration / len(batch))
        else:
            nepoch = int(iteration / len(batch)) + 1
        if optimizer == None:
            optimizer = torch.optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=0.0)
        else:
            pass
        for epoch in range(nepoch):
            # For each training example...
            for i in range(len(batch)):
                # One SGD step
                self.Loss += self.optimizing(batch[i], num_examples_seen, optimizer).item()

                Loss_len += 1
                if int(ITERATION / 100) == 0:
                    time_ = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(time_, ' ', int(100 * num_examples_seen / ITERATION), end='')
                    print('%   완료!!!', end='')
                    print('   loss :', self.Loss / Loss_len)
                    last_loss=self.Loss / Loss_len
                    self.Loss = 0
                    Loss_len = 0
                else:
                    if num_examples_seen % int(ITERATION / 100) == 0:
                        time_ = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        print(time_, ' ', int(100 * num_examples_seen / ITERATION), end='')
                        print('%   완료!!!', end='')
                        print('   loss :', self.Loss / Loss_len)
                        last_loss = self.Loss / Loss_len
                        self.Loss = 0
                        Loss_len = 0
                num_examples_seen += 1
        return last_loss, optimizer


class multi_initialized_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.models = nn.ModuleList([test(input_lang.n_words, output_lang.n_words) for _ in range(Integration_object)])

    def multi_train(self, batch):
        last_losses = []
        view_iteration = int(ITERATION / Integration_number)
        print('view_iteration :', view_iteration)
        fit_opt = None

        for i in range(Integration_number):
            last_losses = []
            min_last_loss = 1000
            optimal_model_index = 0
            optimizers = []
            print('Integration_number :', i + 1)

            for index, model_in in enumerate(self.models):
                print('model :', index + 1)

                if i > 0:
                    checkpoint = torch.load('optimal_model')
                    model_in.load_state_dict(checkpoint['model_state_dict'])
                    fit_opt = torch.optim.Adam(model_in.parameters(), betas=(0.9, 0.98), eps=1e-9,
                                               lr=0.0)  # get new optimiser
                    fit_opt.load_state_dict(checkpoint['optimizer_state_dict'])
                    #가운데 있는 model은 parameter를 바꾸지 않는다.
                    if index!=(Integration_object-1)/2:
                        for param in model_in.parameters():
                            param.data+=torch.randn_like(param.data).uniform_(-1e-3, 1e-3)
                    else:
                        pass
                    model_in.Loss=0

                last_loss, optimizer = model_in.train_with_batch(batch, fit_opt, view_iteration, i * view_iteration + 1)
                last_losses.append(last_loss)
                optimizers.append(optimizer)
                if last_loss < min_last_loss:
                    min_last_loss = last_loss
                    optimal_model_index = index

            torch.save({
                'model_state_dict': self.models[optimal_model_index].state_dict(),
                'optimizer_state_dict': optimizers[optimal_model_index].state_dict()
            }, 'optimal_model')

            print(last_losses)

        return last_losses[int((Integration_object - 1) / 2)]

    def multi_predict(self, batch):
        optimal_model_index = int((Integration_object - 1) / 2)
        predicts = []
        for i in range(len(batch)):
            predicts.append(self.models[optimal_model_index].predict(batch[i][0]))

            if i % 10 == 0:
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(time, ' ', 100 * i / len(batch), '%   저장!!!')

        return predicts


torch.manual_seed(10)
# Train on a small subset of the data to see what happens
model = multi_initialized_model().to(device)

"""""
for parameter in model.parameters():
    print(parameter)
"""""

batch = generate_batch()
print('preprocess done!!!')
last_loss = model.multi_train(batch)
predicts = model.multi_predict(batch)


def save(file_name):
    f = open(file_name, 'w')
    f.write('last loss : %s\n' % last_loss)
    f.write('\n')
    pre_k = 0
    for index, k in enumerate(predicts):
        for j in range(len(k)):
            f.write('input : %s\n' % pairs[:, 0][j + pre_k])
            f.write('result : %s\n' % pairs[:, 1][j + pre_k])
            f.write('predict : %s\n' % [output_lang.index2word[i.item()] for i in k[j] if i != 0])
            f.write('\n')
        pre_k += len(k)
    f.close()
    print("저장 완료!!!")


save('[idea1_last12]view_sentence=' + str(view_sentence_len) + ' batch_size=' + str(Batch_size) + ' iteration=' + str(
    ITERATION) + ' layer1=' + str(Layer1) + '.txt')

torch.save(model.state_dict(), 'saved_model')


