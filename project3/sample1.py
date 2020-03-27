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
import copy


os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # GPU number to use

print('gpu :', torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_default_dtype(torch.float)

print('cudnn :', torch.backends.cudnn.enabled == True)
torch.backends.cudnn.benchmark=True
#####################################################
view_sentence_len = 1024
unknown_number = 5
MAX_LENGTH = 175
MAX_TOKEN=1500
ITERATION=1
hidden_dim=512
Dropout=0.1
#####################################################

w2i={"<Null>" : 0 , "<CLS>" : 1, "<SEP>" : 2,  "<UNKNOWN>" : 3, "<MASK>" : 4}
word_count={}
corpus=[]

f=open('news.en-00001-of-00100', 'r', encoding='utf8')
while True:
    line = f.readline()
    if not line: break
    corpus.append(line.split())

    for i in line.split():
        if i not in w2i:
            w2i[i]=len(w2i)
            word_count[i]=1
        else:
            word_count[i]+=1
f.close()


########training set 만들기###########
temp=[]
correct_dataset=[]
total_pair=0
for sentence in corpus:
    temp.append([w2i[word] for word in sentence])
    if len(temp)==2:
        total_pair+=1
        correct_dataset.append(copy.deepcopy(temp))
        del temp[0]


print('total_pair :', total_pair)

def random_sen():
    return correct_dataset[random.randrange(total_pair)][1]


def random_mask(sen):
    remaked_sen=[]
    masked_index=[]
    for index, word_index in enumerate(sen):
        prob = random.random()
        if prob < 0.15:
            masked_index.append(1)
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                temp=4

            # 10% randomly change token to random token
            elif prob < 0.9:
                temp = random.randrange(len(w2i))

            # 10% randomly change token to current token
            else:
                temp = word_index

        else:
            masked_index.append(0)
            temp=word_index


        remaked_sen.append(temp)


    return remaked_sen, masked_index

pre_X_train=[]
pre_y_train=[]
pre_segment_train=[]
pre_next_sen_label_train=[]
pre_masked_index=[]
for sen_pair in correct_dataset:
    #correct next sentence
    if random.random() > 0.5:
        if len(sen_pair[0]+sen_pair[1])+3>100:
            pass
        else:
            remaked_sen1, masked_id1=random_mask(sen_pair[0])
            remaked_sen2, masked_id2=random_mask(sen_pair[1])
            pre_X_train.append([len(pre_X_train)]+[1]+remaked_sen1+[2]+remaked_sen2+[2])
            pre_y_train.append([1]+sen_pair[0]+[2]+sen_pair[1]+[2])
            pre_segment_train.append([1] * (len(sen_pair[0]) + 2) + [2] * (len(sen_pair[1])+1))
            pre_next_sen_label_train.append([1])
            pre_masked_index.append([0]+masked_id1+[0]+masked_id2+[0])
    else:
        replace_sen=random_sen()
        if len(sen_pair[0]+replace_sen)+3>100:
            pass
        else:
            remaked_sen1, masked_id1 = random_mask(sen_pair[0])
            remaked_sen2, masked_id2 = random_mask(replace_sen)
            pre_X_train.append([len(pre_X_train)]+[1] + remaked_sen1 + [2] + remaked_sen2+[2])
            pre_y_train.append([1] + sen_pair[0] + [2] + replace_sen+[2])
            pre_segment_train.append([1]*(len(sen_pair[0])+2)+[2]*(len(replace_sen)+1))
            pre_next_sen_label_train.append([0])
            pre_masked_index.append([0] + masked_id1 + [0] + masked_id2 + [0])

key = list(reversed(sorted(pre_X_train, key=len)))
k = len(key[0][1:])
max_num_sen = int(MAX_TOKEN / k)
print('k :', k)
print('max_num_sen :', max_num_sen)
num_sen = 0

X_train=[]
y_train=[]
segment_train=[]
next_sen_label_train=[]
masked_index=[]
batch=[]
for input in key:
    padding=[0]*(k-len(input[1:]))
    X_train.append(input[1:]+padding)
    y_train.append(pre_y_train[input[0]]+padding)
    segment_train.append(pre_segment_train[input[0]]+padding)
    next_sen_label_train.append(pre_next_sen_label_train[input[0]])
    masked_index.append(pre_masked_index[input[0]]+padding)
    num_sen += 1
    if num_sen == max_num_sen:
        k = len(key[num_sen][1:])
        max_num_sen = int(MAX_TOKEN / k)
        num_sen = 0
        batch.append((X_train, y_train, segment_train, next_sen_label_train, masked_index))
        X_train = []
        y_train = []
        segment_train = []
        next_sen_label_train=[]
        masked_index=[]

batch.append((X_train, y_train, segment_train, next_sen_label_train, masked_index))

print(batch[0][0])
print(batch[0][1])
print(batch[0][2])
print(batch[0][3])
print(batch[0][4])
