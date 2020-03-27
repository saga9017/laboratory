#transformer : encoder, decoder 6, use residual
# -*- coding: UTF8 -*-

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
from transformers import *
import json


os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # GPU number to use

print('gpu :', torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_default_dtype(torch.float)

print('cudnn :', torch.backends.cudnn.enabled == True)
torch.backends.cudnn.benchmark=True
#####################################################
MAX_LENGTH = 512
MAX_SEN=5
hidden_dim=768
Dropout=0.1
#####################################################



SQuAD_train = []
SQuAD_test = []
#num_classes = 2  # the number of classes

# sentences of training data

with open('SQuAD/train-v1.1.json') as json_file:
    SQuAD_train_dic = json.load(json_file)


for i in SQuAD_train_dic['data']:
    for j in i['paragraphs']:
        for k in j['qas']:
            SQuAD_train.append([k['question'], j['context'], (len(k['question'].split())+k['answers'][0]['answer_start'], len(k['question'].split())+k['answers'][0]['answer_start']+len(k['answers'][0]['text'].split())-1)])



with open('SQuAD/dev-v1.1.json') as json_file:
    SQuAD_test_dic = json.load(json_file)


for i in SQuAD_test_dic['data']:
    for j in i['paragraphs']:
        for k in j['qas']:
            labels=[]
            for l in k['answers']:
                labels.append((len(k['question'].split())+l['answer_start'], len(k['question'].split())+l['answer_start']+len(l['text'].split())-1))
            SQuAD_test.append([k['question'], j['context'], labels])




pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
# Models can return full list of hidden-states & attentions weights at each layer
bert = BertModel.from_pretrained(pretrained_weights,
                                 output_hidden_states=True,
                                 output_attentions=True, force_download=True)


print(SQuAD_test[:10])


X_train=[]
X_train_seg=[]
y_train=[]
for sens in SQuAD_train:
    y_train.append(sens[-1])
    X_train.append([tokenizer.cls_token_id]+tokenizer.encode(sens[0])+[tokenizer.sep_token_id]+tokenizer.encode(sens[1])+[tokenizer.sep_token_id])
    pre_len=len(tokenizer.encode(sens[0]))+2
    X_train_seg.append([0]*pre_len+[1]*(MAX_LENGTH-pre_len))

print('X_train :', len(X_train))
print('y_train :', len(y_train))

X_test=[]
X_test_seg=[]
y_test=[]
for sens in SQuAD_test:
    y_test.append(sens[-1])
    X_test.append([tokenizer.cls_token_id] + tokenizer.encode(sens[0]) + [tokenizer.sep_token_id] + tokenizer.encode(sens[1]) + [tokenizer.sep_token_id])
    pre_len = len(tokenizer.encode(sens[0])) + 2
    X_test_seg.append([0] * pre_len + [1] * (MAX_LENGTH - pre_len))


print('X_test :', len(X_test))
print('X_test_seg :', len(X_test_seg))
print('y_test :', len(y_test))

batch_train=[]

temp_X=[]
temp_X_seg=[]
temp_y=[]
for i in range(len(X_train)):
    if len(X_train[i])<MAX_LENGTH and len(X_train_seg[i])<=MAX_LENGTH and y_train[i][1]<MAX_LENGTH:
        temp_X.append(X_train[i]+[tokenizer.pad_token_id]*(MAX_LENGTH-len(X_train[i])-1)+[tokenizer.sep_token_id])
        temp_X_seg.append(X_train_seg[i])
        temp_y.append(y_train[i])
    else:
        print('error')
    if len(temp_X)==MAX_SEN:
        batch_train.append((temp_X, temp_X_seg, temp_y))
        temp_X = []
        temp_X_seg = []
        temp_y = []

if temp_X!=[]:
    batch_train.append((temp_X, temp_X_seg, temp_y))

batch_test = []

temp_X = []
temp_X_seg = []
temp_y = []
for i in range(len(X_test)):
    flag=True
    for s,e  in y_test[i]:
        if e >= MAX_LENGTH:
            flag=False
    if len(X_test[i]) < MAX_LENGTH and len(X_test_seg[i])<=MAX_LENGTH and flag:
        temp_X.append(X_test[i] + [tokenizer.pad_token_id] * (MAX_LENGTH - len(X_test[i])-1)+[tokenizer.sep_token_id])
        temp_X_seg.append(X_test_seg[i])
        temp_y.append(y_test[i])
    else:
        print('error')
    if len(temp_X) == MAX_SEN:
        batch_test.append((temp_X, temp_X_seg, temp_y))
        temp_X = []
        temp_X_seg = []
        temp_y = []

if temp_X!=[]:
    batch_test.append((temp_X, temp_X_seg, temp_y))


#################################################### Module ############################################################


class Bert(nn.Module):
    def __init__(self):
        # Assign instance variables
        super(Bert, self).__init__()
        self.bert = copy.deepcopy(bert)

    def forward(self, x, seg):
        outputs= self.bert(x, token_type_ids=seg)
        last_encoder_output = outputs[0]
        return last_encoder_output

class NextSentencePrediction(nn.Module):


    def __init__(self, hidden):

        super().__init__()
        self.start = nn.Linear(hidden, 1)
        self.end = nn.Linear(hidden, 1)
        nn.init.xavier_normal_(self.start.weight)
        nn.init.xavier_normal_(self.end.weight)

    def forward(self, x):
        return self.start(x), self.end(x)

class BERTLM(nn.Module):


    def __init__(self):

        super().__init__()
        self.bert = Bert()
        self.next_sentence = NextSentencePrediction(768)
        self.Loss = 0
        self.softmax = nn.Softmax(-1)


    def forward(self, x, seg):
        x = self.bert(x, seg)
        return self.next_sentence(x)

    def bptt(self, x, seg, y):  # (batch_size, out_len)
        model.train()
        start, end = self.forward(x, seg)
        softmax_s = self.softmax(start.squeeze(-1))
        softmax_e = self.softmax(end.squeeze(-1))
        loss_s = torch.mean(-torch.log(softmax_s.gather(1, y[: ,0].unsqueeze(-1))))
        loss_e = torch.mean(-torch.log(softmax_e.gather(1, y[: ,1].unsqueeze(-1))))
        loss=loss_s+loss_e
        return loss

    def evaluation(self, batch_test):
        model.eval()
        score = 0
        sample_num = 0
        for i in range(len(batch_test)):
            start, end = self.forward(torch.tensor(batch_test[i][0]).to(device), torch.tensor(batch_test[i][1]).to(device))
            start=self.softmax(start).squeeze(-1)
            end=self.softmax(end).squeeze(-1)
            max=0
            optimal_pair=()
            for batch in range(len(start)):
                threshold=batch_test[i][1][batch].index(1)
                values_s, indices_s = torch.topk(start[batch], 20)
                values_e, indices_e = torch.topk(end[batch], 20)
                for j in indices_s:
                    for k in indices_e:
                        if j<threshold:
                            continue
                        if k<j:
                            continue
                        point=start[batch][j]+end[batch][k]
                        if point >max:
                            max=point
                            optimal_pair=(j.item(),k.item())
                if optimal_pair in batch_test[i][2][batch]:
                    score+=1
            sample_num += len(batch_test[i])

            time_ = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if i%10==0:
                print(time_, ' ', (100 * i)/ len(batch_test))

        print('score :', score / sample_num)

        # Performs one step of SGD.

    def numpy_sdg_step(self, batch, optimizer):
        # Calculate the gradients

        optimizer.zero_grad()
        loss = self.bptt(torch.tensor(batch[0]).to(device).long(), torch.tensor(batch[1]).to(device).long(),
                         torch.tensor(batch[2]).to(device).long())

        loss.backward()

        optimizer.step()
        # optimizer.param_groups[0]['lr'] = lrate

        return loss

    def train_with_batch(self, batch, batch_test):
        # We keep track of the losses so we can plot them later
        num_examples_seen = 1
        Loss_len = 0
        last_loss = 0
        #nepoch = int(ITERATION / len(batch))+1
        nepoch=10
        print('epoch :', nepoch)
        optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), eps=1e-9, lr=2e-5)
        for epoch in range(nepoch):
            # For each training example...
            for i in range(len(batch)):
                # One SGD step
                self.Loss += self.numpy_sdg_step(batch[i], optimizer).item()

                Loss_len += 1
                if num_examples_seen == len(batch) * nepoch:
                    last_loss = self.Loss / Loss_len
                else:
                    if int(len(batch) * nepoch / 100) == 0:
                        time_ = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        print(time_, ' ', int(100 * num_examples_seen / (len(batch) * nepoch)), end='')
                        print('%   완료!!!', end='')
                        print('   loss :', self.Loss / Loss_len)
                        self.Loss = 0
                        Loss_len = 0
                    else:
                        if num_examples_seen % int(len(batch) * nepoch / 100) == 0:
                            time_ = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            print(time_, ' ', int(100 * num_examples_seen / (len(batch) * nepoch)), end='')
                            print('%   완료!!!', end='')
                            print('   loss :', self.Loss / Loss_len)
                            self.Loss = 0
                            Loss_len = 0
                num_examples_seen += 1
            self.evaluation(batch_test)
        return last_loss



torch.manual_seed(10)
# Train on a small subset of the data to see what happens

model = BERTLM().to(device)
last_loss = model.train_with_batch(batch_train, batch_test)

print(last_loss)
