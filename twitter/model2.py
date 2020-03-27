"""""""""""""""""""""""""""
using pre-trained bert, using cls for prediction, fine-tuning, using text + location

"""""""""""""""""""""""""""

import sys, os
from datetime import datetime
import numpy as np
import random
import copy
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math
from transformers import *
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

print('gpu :', torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import pickle

Dropout = 0.1
hidden_dim = 768
MAX_LENGTH = 450

pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)




# bert = BertModel.from_pretrained(pretrained_weights,
#                                  output_hidden_states=True,
#                                  output_attentions=True, force_download=True)

with open('bert.pickle', 'rb') as f:
    bert = pickle.load(f)

class Bert(nn.Module):
    def __init__(self):
        # Assign instance variables
        super(Bert, self).__init__()
        self.bert = copy.deepcopy(bert)

    def forward(self, x, attention_mask, token_type_ids):

        outputs= self.bert(x, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_encoder_output = outputs[0]
        return last_encoder_output



class FCLayer(nn.Module):
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




class BERTLM_new(nn.Module):
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
        self.next_sentence = FCLayer(768)

        self.Loss = 0
        self.softmax = nn.Softmax(-1)
        self.criteria=nn.CrossEntropyLoss()

    def forward(self, batch_text, batch_seg, dropout):
        attention_mask=(batch_text!=tokenizer.pad_token_id)
        x = self.bert(batch_text, attention_mask=attention_mask.float(), token_type_ids=batch_seg)
        return self.next_sentence(x)

    def bptt(self, batch_loc_text, batch_seg, label, dropout):  # (batch_size, out_len)
        result = self.forward(batch_loc_text, batch_seg, dropout)
        loss=self.criteria(result, label)
        return loss

        # Performs one step of SGD.

    def numpy_sdg_step(self, batch, optimizer, lrate, dropout):
        # Calculate the gradients

        optimizer.zero_grad()


        loss = self.bptt(torch.tensor(batch[0]).to(device).long(), torch.tensor(batch[1]).to(device).long(), torch.tensor(batch[2]).to(device).long(), dropout)

        loss.backward()
        optimizer.step()
        # optimizer.param_groups[0]['lr'] = lrate

        return loss

    def train_with_batch(self, info, batch_val, batch_seg_, batch_label_, optimizer):
        val_precision = []

        Loss_len = 0
        num_examples_seen = 1
        nepoch = 1000
        print('training epoch :', nepoch)
        print('lengh of batch :', int(len(info[0]) / 10))
        for epoch in range(nepoch):
            epoch=epoch
            #info=suffle(info)
            print("에폭", epoch + 1)
            batch_train, batch_seg, batch_label = generate_batch(info[0], info[1], info[2])

            for i in range(len(batch_train)):
                self.train()
                batch_text, batch_s, batch_y = batch_train[i], batch_seg[i], batch_label[i]

                # print(np.shape(batch_know)) # [batch_size, cat_num]
                # 3factors or 4factors
                lrate = math.pow(64, -0.5) * min(math.pow(num_examples_seen, -0.5), num_examples_seen * math.pow(4000,
                                                                                                                 -1.5))  # warm up step : default 4000
                loss = self.numpy_sdg_step((batch_text, batch_s, batch_y), optimizer, lrate, True)
                self.Loss += loss.item()
                Loss_len += 1

                if num_examples_seen % 500 == 0:  # origin = int(batch_len * nepoch / 100)
                    time_ = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(time_, ' ', int(100 * num_examples_seen / (len(batch_train) * nepoch)), end='')
                    print('%   완료!!!', end='')
                    print('   loss :', self.Loss / Loss_len)  # , '   lr :', lrate)
                    self.Loss = 0
                    Loss_len = 0

                num_examples_seen += 1

            print('Epoch', epoch + 1, 'completed out of', nepoch)
            # valid set : 8104개
            score=0
            total=0
            for i in range(len(batch_val)):
                self.eval()
                batch_text, batch_s, batch_y,  = batch_val[i], batch_seg_[i], batch_label_[i]
                val_prob = self.softmax(
                    self.forward(torch.tensor(batch_text).to(device).long(), torch.tensor(batch_s).to(device).long(), False))

                # if last_function() == "softmax":
                y_pred = np.argmax(val_prob.detach().cpu().numpy(), axis=1)
                score+=np.sum(y_pred==batch_y)
                total+=len(batch_y)

            print("Epoch:", (epoch + 1), "val_precision:", float(score)/total)
            val_precision.append(float(score)/total)

        return val_precision




def max_length(list):
    max = 0
    for i in list:
        if max < len(i):
            max = len(i)

    return max

def generate_batch(corpus, seg, label):
    corpus_len = len(corpus)
    iters = int(corpus_len / 10)
    tmp_batch = []
    tmp_battch_seg = []
    tmp_batch_label = []
    for iter in range(iters):
        tmp = corpus[iter*10:(iter+1)*10]
        tmp_seg=seg[iter*10:(iter+1)*10]
        tmp2 = []
        tmp_seg2=[]
        max_len = 0
        for sentence in tmp:
            if max_len < len(sentence):
                max_len = len(sentence)
        for sentence, seg_ in zip(tmp, tmp_seg):
            sentence = sentence + [tokenizer.pad_token_id] *(max_len -len(sentence))
            seg_ = seg_ + [1]*(max_len -len(seg_))
            tmp2.append(sentence)
            tmp_seg2.append(seg_)
        tmp_batch.append(tmp2)
        tmp_battch_seg.append(tmp_seg2)
        tmp_batch_label.append(label[iter*10:(iter+1)*10])

    tmp_batch_label.append(label[iters*10:])
    return tmp_batch, tmp_battch_seg, tmp_batch_label


def suffle(info):
    sample_list = [i for i in range(len(info[0]))]
    random.shuffle(sample_list)
    new_info_0 = []
    new_info_1 = []
    new_info_2 = []
    for i in sample_list:
        new_info_0.append(info[0][i])
        new_info_1.append(info[1][i])
        new_info_2.append(info[2][i])

    new_info = []
    new_info.append(new_info_0)
    new_info.append(new_info_1)
    new_info.append(new_info_2)

    return new_info

if __name__ == '__main__':

    with open('textnextract.pickle', 'rb') as f:
        multi_modal_dic = pickle.load(f)

    dic_list = list(multi_modal_dic)
    random.shuffle(dic_list)
    new_val_text_list = []
    val_label = []
    val_seg=[]
    for key in dic_list[:3000]:
        text, label, extract = multi_modal_dic[key]
        tmp_extract=tokenizer.encode(extract)
        tmp_sen = tmp_extract + tokenizer.encode(text)[1:]
        tmp_seg =[0]*len(tmp_extract)+[1]*(len(tmp_sen)-len(tmp_extract))
        #tmp_sen = tokenizer.encode(text)
        new_val_text_list.append(tmp_sen)
        val_label.append(label)
        val_seg.append(tmp_seg)

    new_test_text_list = []
    test_label = []
    test_seg = []
    for key in dic_list[3000:6000]:
        text, label, extract = multi_modal_dic[key]
        tmp_extract = tokenizer.encode(extract)
        tmp_sen = tmp_extract + tokenizer.encode(text)[1:]
        tmp_seg = [0] * len(tmp_extract) + [1] * (len(tmp_sen) - len(tmp_extract))
        #tmp_sen = tokenizer.encode(text)
        new_test_text_list.append(tmp_sen)
        test_label.append(label)
        test_seg.append(tmp_seg)

    new_train_text_list = []
    train_label = []
    train_seg = []
    for key in dic_list[6000:]:
        text, label, extract = multi_modal_dic[key]
        tmp_extract = tokenizer.encode(extract)
        tmp_sen = tmp_extract + tokenizer.encode(text)[1:]
        tmp_seg = [0] * len(tmp_extract) + [1] * (len(tmp_sen) - len(tmp_extract))
        #tmp_sen = tokenizer.encode(text)
        new_train_text_list.append(tmp_sen)
        train_label.append(label)
        train_seg.append(tmp_seg)



    batch_val, batch_seg, batch_label =generate_batch(new_val_text_list, val_seg, val_label)


    model = BERTLM_new().to(device)
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), eps=1e-9, lr=1e-5)
    print("starts training...")


    val_precision = model.train_with_batch((new_train_text_list, train_seg, train_label), batch_val, batch_seg, batch_label, optimizer)


