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
batch_size=512
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
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


def sigmoid(z):
    return 1 / (1 + torch.exp(-z))


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print('pair:', random.choice(pairs))

pairs = np.array(pairs)

candidate=[]
f_can=open('candidate.txt', 'r', encoding='utf8')
lines = f_can.readlines()
for line in lines:
    candidate.append(line.split('\n')[0])
f_can.close()
print(candidate)

can_vocab=[]
for sen in candidate:
    for word in sen.split():
        if word not in can_vocab:
            can_vocab.append(word)

print(can_vocab)

train_len=0
f_train=open('translation_train.txt', 'w')
for sen1, sen2 in pairs:
    flag=True
    if sen1 not in candidate:
        for word in sen1.split():
            if word not in can_vocab:
                flag=False

        if flag==True:
            f_train.write(sen1)
            f_train.write('\t')
            f_train.write(sen2)
            f_train.write('\n')
            train_len+=1
f_train.close()


test_len=0
f_test=open('translation_test.txt', 'w')
for sen1, sen2 in pairs:
    if sen1 in candidate:
        f_test.write(sen1)
        f_test.write('\t')
        f_test.write(sen2)
        f_test.write('\n')
        test_len+=1
f_test.close()

print('train set :', train_len)
print('test set :', test_len)