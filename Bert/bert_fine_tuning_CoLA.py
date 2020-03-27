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
from transformers import *
import csv


os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # GPU number to use

print('gpu :', torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_default_dtype(torch.float)

print('cudnn :', torch.backends.cudnn.enabled == True)
torch.backends.cudnn.benchmark=True
#####################################################
MAX_LENGTH = 100
MAX_SEN=15
hidden_dim=768
Dropout=0.1
#####################################################



CoLA_train = []
CoLA_test = []
CoLA_eval = []
num_classes = 2  # the number of classes

# sentences of training data
with open('glue_data/CoLA/in_domain_train.tsv', 'r', encoding='latin-1') as f:
    for row in f.readlines():
        s=' '.join(row.split('\t')[3].replace('(', '').replace(')', '').split())
        label=row.split('\t')[1]
        CoLA_train.append([s, int(label)])

# sentences of test data
with open('glue_data/CoLA/original/tokenized/out_of_domain_dev.tsv', 'r', encoding='latin-1') as f:
    for row in f.readlines():
        s = ' '.join(row.split('\t')[3].replace('(', '').replace(')', '').split())
        label = row.split('\t')[1]
        CoLA_test.append([s, int(label)])



with open('glue_data/CoLA/cola_out_of_domain_test.tsv', 'r', encoding='latin-1') as f:
    for row in f.readlines():
        s = ' '.join(row.split('\t')[1].replace('(', '').replace(')', '').split())
        id = row.split('\t')[0]
        CoLA_eval.append([s, id])



pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
# Models can return full list of hidden-states & attentions weights at each layer
bert = BertModel.from_pretrained(pretrained_weights,
                                 output_hidden_states=True,
                                 output_attentions=True, force_download=True)



print(CoLA_test[:10])

X_train=[]
y_train=[]
for sen in CoLA_train:
    y_train.append(sen[1])
    X_train.append([tokenizer.cls_token_id]+tokenizer.encode(sen[0])+[tokenizer.sep_token_id])

print('X_train :', len(X_train))
print('y_train :', len(y_train))

X_test=[]
y_test=[]
for sen in CoLA_test:
    y_test.append(sen[1])
    X_test.append([tokenizer.cls_token_id]+tokenizer.encode(sen[0])+[tokenizer.sep_token_id])


X_eval=[]
y_eval=[]
for sen in CoLA_eval:
    y_eval.append(sen[1])
    X_eval.append([tokenizer.cls_token_id]+tokenizer.encode(sen[0])+[tokenizer.sep_token_id])


print('X_test :', len(X_test))
print('y_test :', len(y_test))

batch_train=[]

temp_X=[]
temp_X_seg=[]
temp_y=[]
for i in range(len(X_train)):
    if len(X_train[i])<=100:
        temp_X.append(X_train[i]+[tokenizer.pad_token_id]*(100-len(X_train[i])-1)+[tokenizer.sep_token_id])
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
    if len(X_test[i]) <= 100:
        temp_X.append(X_test[i] + [tokenizer.pad_token_id] * (100 - len(X_test[i])-1)+[tokenizer.sep_token_id])
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


batch_eval = []

temp_X = []
temp_X_seg = []
temp_y = []
for i in range(len(X_eval)):
    if len(X_eval[i]) <= 100:
        temp_X.append(X_eval[i] + [tokenizer.pad_token_id] * (100 - len(X_eval[i])-1)+[tokenizer.sep_token_id])
        temp_y.append(y_eval[i])
    else:
        print('error')
    if len(temp_X) == MAX_SEN:
        batch_eval.append((temp_X, temp_X_seg, temp_y))
        temp_X = []
        temp_X_seg = []
        temp_y = []

if temp_X!=[]:
    batch_eval.append((temp_X, temp_X_seg, temp_y))

print(batch_eval)
#################################################### Module ############################################################

"""""
class bert_embedding(nn.Module):
    def __init__(self):
        super(bert_embedding, self).__init__()

    def forward(self, x):
        return x
"""

class Bert(nn.Module):
    def __init__(self):
        # Assign instance variables
        super(Bert, self).__init__()
        self.bert = copy.deepcopy(bert)

    def forward(self, x):
        outputs= self.bert(x)
        last_encoder_output = outputs[0]
        return last_encoder_output

class NextSentencePrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        nn.init.xavier_normal_(self.linear.weight)


    def forward(self, x):
        return self.linear(x[:, 0])

class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = Bert()
        self.next_sentence = NextSentencePrediction(768)
        self.Loss = 0
        self.softmax = nn.Softmax(-1)


    def forward(self, x):
        x = self.bert(x)
        return self.next_sentence(x)

    def bptt(self, x, seg, y):  # (batch_size, out_len)
        next_sentence = self.forward(x)
        softmax=self.softmax(next_sentence)
        loss = -torch.log(softmax.gather(1, y.unsqueeze(-1)))
        loss=torch.mean(loss)
        return loss

    def evaluation(self, batch_test):
        model.eval()
        score = 0
        sample_num = 0
        for i in range(len(batch_test)):
            next_sentence = self.forward(torch.tensor(batch_test[i][0]).to(device))
            matching = (torch.argmax(next_sentence, dim=1) == torch.tensor(batch_test[i][2]).to(device))
            score += torch.sum(matching).float()
            sample_num += len(matching)

        print('score :', score / sample_num)

    def save_evaluation(self, batch_eval, filename):
        model.eval()
        result=[]
        for i in range(len(batch_eval)):
            next_sentence = self.forward(torch.tensor(batch_eval[i][0]).to(device))
            for index, j in enumerate(batch_eval[i][2]):
                result.append((j, torch.argmax(next_sentence, dim=1)[index].item()))

        csvfile=open(filename, 'w')
        csvwriter=csv.writer(csvfile)
        for row in result:
            csvwriter.writerow(row)

        csvfile.close()

        # Performs one step of SGD.

    def numpy_sdg_step(self, batch, optimizer):
        # Calculate the gradients

        optimizer.zero_grad()
        loss = self.bptt(torch.tensor(batch[0]).to(device).long(), torch.tensor(batch[1]).to(device).long(),
                         torch.tensor(batch[2]).to(device).long())

        loss.backward()
        """
        for p in model.parameters():
            p.data.clamp_(-5, 5)
        """
        optimizer.step()
        # optimizer.param_groups[0]['lr'] = lrate

        return loss

    def train_with_batch(self, batch, batch_test, batch_eval):
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
            self.save_evaluation(batch_eval, 'CoLA_test_result'+str(epoch)+'.csv')
        return last_loss



torch.manual_seed(10)
# Train on a small subset of the data to see what happens

model = BERTLM().to(device)
last_loss = model.train_with_batch(batch_train, batch_test, batch_eval)

print(last_loss)

