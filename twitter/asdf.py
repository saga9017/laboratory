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

from project3_data.result_calculator import *
import torch.nn.utils as torch_utils

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

print('gpu :', torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import pickle

import tensorflow as tf
from project3_data.config import *

Dropout = 0.1
hidden_dim = 768
MAX_LENGTH = 450


# ★ oversampling
# def train_generate_batch(t_maxlen, l_maxlen, hashtag_size, hashtagCount, hashtagCount_saved, tag_used):
#     batch_img = np.zeros((batch_size(), 1, feat_dim(), w(), w()))
#     batch_text = np.zeros((batch_size(), t_maxlen))
#     batch_loc = np.zeros((batch_size(), l_maxlen))
#     batch_y = np.zeros(batch_size())
#     truth_hashtag = []
#     text_start = []
#     epoch_finish = False
#     for i in range(batch_size()):
#
#         hashtag_choice = random.randrange(0, hashtag_size)
#         data_choice = random.randrange(0, len(hashtagCount[hashtag_choice]))
#         # print(hashtag_choice, data_choice, "/", len(hashtagCount[hashtag_choice]))
#         # while True:
#         #     hashtag_choice = random.randrange(0, hashtag_size)
#         #     if tag_used[hashtag_choice] == False:
#         #         data_choice = random.randrange(0, len(hashtagCount[hashtag_choice]))
#         #         break
#
#         data_index = hashtagCount[hashtag_choice][data_choice]
#         batch_img[i] = train_data[0][data_index]
#         batch_text[i] = train_data[1][data_index]
#         batch_loc[i] = train_data[2][data_index]
#         batch_y[i] = hashtag_choice
#         truth_hashtag.append(train_data[3][data_index])
#
#         allzero = False
#         for q, j in enumerate(batch_text[i]):
#             if int(j) != 0:
#                 text_start.append(q)
#                 allzero = True
#                 break
#         if allzero == False: text_start.append(0)
#
#         del hashtagCount[hashtag_choice][data_choice]
#         if len(hashtagCount[hashtag_choice]) == 0:
#             tag_used[hashtag_choice] = True
#             hashtagCount[hashtag_choice] = copy.deepcopy(hashtagCount_saved[hashtag_choice])
#             if np.all(tag_used) == True:
#                 print("다썼다!")
#                 tag_used = [False for g in range(hashtag_size)]
#                 epoch_finish = True
#                 break
#
#     return batch_img, batch_text, batch_loc, batch_y, epoch_finish, truth_hashtag, text_start, tag_used, hashtagCount

# ★ shuffle batch
# def generate_batch(which, pnt, y_cnt, t_maxlen, l_maxlen, finish):
#     batch_img = np.zeros((batch_size(), 1, feat_dim(), w(), w()))
#     batch_text = np.zeros((batch_size(), t_maxlen))
#     batch_loc = np.zeros((batch_size(), l_maxlen))
#     batch_y = np.zeros(batch_size())
#     batch_cnt = 0
#     truth_hashtag = []
#     shuffle = list(range(batch_size()))
#     random.shuffle(shuffle)
#     while True:
#         text_start = []
#         if which == "train":
#             hashend = len(train_data[3][pnt])
#             datalen = len(train_data[0])
#             batch_img[shuffle[batch_cnt]] = train_data[0][pnt]
#             batch_text[shuffle[batch_cnt]] = train_data[1][pnt]
#             batch_loc[shuffle[batch_cnt]] = train_data[2][pnt]
#             batch_y[shuffle[batch_cnt]] = train_data[3][pnt][y_cnt]
#             truth_hashtag.append(train_data[3][pnt])
#         elif which == "validation":
#             hashend = len(val_data[3][pnt])
#             datalen = len(val_data[0])
#             batch_img[shuffle[batch_cnt]] = val_data[0][pnt]
#             batch_text[shuffle[batch_cnt]] = val_data[1][pnt]
#             batch_loc[shuffle[batch_cnt]] = val_data[2][pnt]
#             batch_y[shuffle[batch_cnt]] = val_data[3][pnt][y_cnt]
#             truth_hashtag.append(val_data[3][pnt])
#         else:
#             hashend = len(test_data[3][pnt])
#             datalen = len(test_data[0])
#             batch_img[shuffle[batch_cnt]] = test_data[0][pnt]
#             batch_text[shuffle[batch_cnt]] = test_data[1][pnt]
#             batch_loc[shuffle[batch_cnt]] = test_data[2][pnt]
#             batch_y[shuffle[batch_cnt]] = test_data[3][pnt][y_cnt]
#             truth_hashtag.append(test_data[3][pnt])
#
#         allzero = False
#         for i, j in enumerate(batch_text[shuffle[batch_cnt]]):
#             if int(j) != 0:
#                 text_start.append(i)
#                 allzero = True
#                 break
#         if allzero == False: text_start.append(0)
#
#         # print("------------------------------------------")
#         # print("input text:")
#         # for i in batch_text[batch_cnt]:
#         #     textnum = int(i)
#         #     if textnum != 0:
#         #         print(vocabulary_inv[textnum], end=" ")
#         # print("\ninput loc:")
#         # for i in batch_loc[batch_cnt]:
#         #     locnum = int(i)
#         #     if locnum != 0:
#         #         print(vocabulary_inv[locnum], end=" ")
#         # print("\nTrue hashtag:")
#         # for i in truth_hashtag[batch_cnt]:
#         #     print(hashtagVoc_inv[int(i)], end="||")
#         # print()
#         y_cnt += 1
#         batch_cnt += 1
#
#         if y_cnt == hashend:
#             y_cnt = 0
#             pnt += 1
#             if pnt == datalen:
#                 pnt = 0
#                 finish = True
#
#         if finish or batch_cnt == batch_size(): break
#     return batch_img, batch_text, batch_loc, batch_y, pnt, y_cnt, finish, truth_hashtag, text_start


def load_k():
    k_train, k_val, k_test = [], [], []
    with open("project3_data/txt/e5_train_insta_textonly.txt", "r") as f:
        while True:
            line = f.readline()
            if not line: break
            line = line.split()
            for i in range(len(line)):
                line[i] = int(line[i])
            k_train.append(line)

    with open("project3_data/txt/e5_val_insta_textonly.txt", "r") as f:
        while True:
            line = f.readline()
            if not line: break
            line = line.split()
            for i in range(len(line)):
                line[i] = int(line[i])
            k_val.append(line)

    with open("project3_data/txt/e5_test_insta_textonly.txt", "r") as f:
        while True:
            line = f.readline()
            if not line: break
            line = line.split()
            for i in range(len(line)):
                line[i] = int(line[i])
            k_test.append(line)

    return k_train, k_val, k_test


"""""
def generate_batch(which, pnt, y_cnt, t_maxlen, l_maxlen, finish):
    batch_img = np.zeros((batch_size(), 1, feat_dim(), w(), w()))
    batch_text = np.zeros((batch_size(), t_maxlen))
    batch_loc = np.zeros((batch_size(), l_maxlen))
    batch_know = np.zeros((batch_size(), cat_num()))
    batch_y = np.zeros(batch_size())
    batch_cnt = 0
    truth_hashtag = []

    while True:
        if which == "train":
            hashend = len(train_data[3][pnt])
            datalen = len(train_data[0])
            batch_img[batch_cnt] = train_data[0][pnt]
            batch_text[batch_cnt] = train_data[1][pnt]
            batch_loc[batch_cnt] = train_data[2][pnt]
            batch_y[batch_cnt] = train_data[3][pnt][y_cnt]
            truth_hashtag.append(train_data[3][pnt])
            batch_know[batch_cnt] = train_data[4][pnt]
        elif which == "validation":
            hashend = len(val_data[3][pnt])
            datalen = len(val_data[0])
            batch_img[batch_cnt] = val_data[0][pnt]
            batch_text[batch_cnt] = val_data[1][pnt]
            batch_loc[batch_cnt] = val_data[2][pnt]
            batch_y[batch_cnt] = val_data[3][pnt][y_cnt]
            truth_hashtag.append(val_data[3][pnt])
            batch_know[batch_cnt] = val_data[4][pnt]
        else:
            hashend = len(test_data[3][pnt])
            datalen = len(test_data[0])
            batch_img[batch_cnt] = test_data[0][pnt]
            batch_text[batch_cnt] = test_data[1][pnt]
            batch_loc[batch_cnt] = test_data[2][pnt]
            batch_y[batch_cnt] = test_data[3][pnt][y_cnt]
            truth_hashtag.append(test_data[3][pnt])
            batch_know[batch_cnt] = test_data[4][pnt]

        y_cnt += 1
        batch_cnt += 1

        if y_cnt == hashend:
            y_cnt = 0
            pnt += 1
            if pnt == datalen:
                pnt = 0
                finish = True

        if finish or batch_cnt == batch_size(): break
    return batch_img, batch_text, batch_loc, batch_know, batch_y, pnt, y_cnt, finish, truth_hashtag
"""""


class Norm(nn.Module):
    def __init__(self):
        super(Norm, self).__init__()
        self.gamma=Parameter(torch.tensor(1.0))
        self.beta = Parameter(torch.tensor(0.0))
    def forward(self, X):
        mu = torch.mean(X, dim=2)
        var = torch.var(X, dim=2)
        X_norm = torch.div(X - mu.view(X.shape[0], X.shape[1], 1), torch.sqrt(var.view(X.shape[0], X.shape[1], 1) + 1e-8))
        out = self.gamma * X_norm + self.beta
        return out

class Multi_head_attention(nn.Module):
    def __init__(self, hidden_dim=300, hidden_dim_=512, dropout=Dropout):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.softmax = nn.Softmax(-1)
        self.layerNorm_add_Norm = Norm()

        self.w_qs = nn.Linear(hidden_dim, hidden_dim_)
        self.w_ks = nn.Linear(hidden_dim, hidden_dim_)
        self.w_vs = nn.Linear(hidden_dim, hidden_dim_)
        self.w_os = nn.Linear(hidden_dim_, hidden_dim)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (hidden_dim)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (hidden_dim)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (hidden_dim)))
        nn.init.xavier_normal_(self.w_os.weight)

    def forward(self, en, de, mask, dropout=False):  # x : (input_len, hidden_dim)
        d_k = de.shape[-1]
        len_d0, len_d1=de.shape[0], de.shape[1]
        len_e0, len_e1=en.shape[0], en.shape[1]


        q = self.w_qs(de).view(len_d0, len_d1, -1, 8).permute(3, 0, 1, 2)
        k = self.w_ks(en).view(len_e0, len_e1, -1, 8).permute(3, 0, 2, 1)
        v = self.w_vs(en).view(len_e0, len_e1, -1, 8).permute(3, 0, 1, 2)

        e = torch.matmul(q, k) / math.sqrt(d_k)
        masked_e = e.masked_fill(mask, -1e10)
        alpha = self.softmax(masked_e)  # (output_len, input_len)
        if dropout==True:
            alpha = self.dropout(alpha)
        head3 = torch.matmul(alpha, v)

        a = torch.cat((head3[0], head3[1], head3[2], head3[3], head3[4],
                       head3[5], head3[6], head3[7]), 2)

        result = self.w_os(a)
        result=self.layerNorm_add_Norm(result+de)
        return result  # (output_len, hidden)

def gelu(x):
  """Gaussian Error Linear Unit.
  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.
  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + torch.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))))
  return x * cdf

class FFN(nn.Module):  # feed forward network   x : (batch_size, input_len, hidden)
    def __init__(self, hidden_dim=300, dropout=Dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layerNorm_add_Norm = Norm()

        self.fc1 = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.fc2 = nn.Linear(4*hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)


    def forward(self, x, dropout=False):
        output = self.fc1(x) # (batch_size, input_len, 4*hidden)
        if dropout==True:
            output = self.dropout(gelu(output))  # (batch_size, input_len, 4*hidden)
        else:
            output = gelu(output)
        output = self.fc2(output)  # (batch_size, input_len, hidden
        output=self.layerNorm_add_Norm(output+x)
        return output

##################################################### Sub layer ########################################################

class Encoder_layer(nn.Module):
    def __init__(self, hidden_dim=300, hidden_dim_=512, dropout=Dropout):  # default=512
        # Assign instance variables
        super().__init__()
        self.multi_head_self_attention=Multi_head_attention(hidden_dim, hidden_dim_, dropout)
        self.ffn=FFN(hidden_dim, dropout)


        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)


    def forward(self, x, mask, non_pad_mask, dropout=False):
        if dropout==True:
            output=self.dropout_1(self.multi_head_self_attention(x, x, mask, dropout=True))
            output=output.masked_fill(non_pad_mask==0, 0)
            output=self.dropout_2(self.ffn(output, dropout=True))
            output=output.masked_fill(non_pad_mask==0, 0)
        else:
            output = self.multi_head_self_attention(x, x, mask)
            output = output.masked_fill(non_pad_mask == 0, 0)
            output = self.ffn(output)
            output = output.masked_fill(non_pad_mask == 0, 0)
        return output


#################################################### Layer ##############################################################


class Transformer_new(nn.Module):
    def __init__(self, hidden_dim=hidden_dim, hidden_dim_=hidden_dim):  # default=512
        # Assign instance variables
        super().__init__()
        self.hidden = hidden_dim

        self.segment_embed = nn.Embedding(3, hidden_dim)
        self.sequence_embed = Parameter(torch.randn(512+(5+1)+(49+1), hidden_dim))

        self.know_embed=nn.Embedding(540, hidden_dim)

        #weight initialization

        self.segment_embed.weight.data.uniform_(-0.01, 0.01)
        self.know_embed.weight.data.uniform_(-0.01, 0.01)
        nn.init.xavier_normal_(self.sequence_embed)


        self.encoder1=Encoder_layer(hidden_dim, hidden_dim_)
        self.encoder2 = Encoder_layer(hidden_dim, hidden_dim_)
        self.encoder3 = Encoder_layer(hidden_dim, hidden_dim_)
        #self.encoder4 = Encoder_layer(hidden_dim, hidden_dim_)
        #self.encoder5 = Encoder_layer(hidden_dim, hidden_dim_)
        #self.encoder6 = Encoder_layer(hidden_dim, hidden_dim_)

    def input_embedding(self, batch_text):  # x: (batch, input_len, )

        b_size, s_len, _=batch_text.shape
        mixed=batch_text
        segment=torch.tensor([2]*s_len).to(device)

        return mixed+self.segment_embed(segment)+self.sequence_embed[:mixed.shape[1]].unsqueeze(0).repeat(b_size, 1, 1)  # (input_len, hidden_dim)

    def forward(self, batch_text, attention_mask, dropout):
        b_size, s_len, _=batch_text.shape
        x1= self.input_embedding(batch_text)  # (input_len, hidden)
        ###################################### make non_pad_mask   #####################################################
        if device.type=='cpu':
            margin=torch.tensor([1] * (x1.shape[1]-s_len)).repeat(b_size, 1).byte().to(device)
        else:
            margin = torch.tensor([1] * (x1.shape[1] - s_len)).repeat(b_size, 1).bool().to(device)
        non_pad_mask = torch.cat([margin, attention_mask], dim=1).unsqueeze(-1).repeat(1, 1, self.hidden).float()
        mask_e = (torch.cat([margin, attention_mask], dim=1).unsqueeze(1).repeat(1, x1.shape[1], 1)==0)


        ################################################################################################################
        x1=x1.masked_fill(non_pad_mask==0, 0)
        ########################################################
        x2 = self.encoder1(x1, mask_e, non_pad_mask, dropout=dropout)
        x3 = self.encoder2(x2 + x1, mask_e, non_pad_mask, dropout=dropout)
        x4 = self.encoder3(x3 + x2 + x1, mask_e, non_pad_mask, dropout=dropout)
        #x5 = self.encoder4(x4 + x3 + x2 + x1, mask_e, non_pad_mask, dropout=dropout)
        #x6 = self.encoder5(x5 + x4 + x3 + x2 + x1, mask_e, non_pad_mask, dropout=dropout)
        #x7 = self.encoder6(x6 + x5 + x4 + x3 + x2 + x1, mask_e, non_pad_mask, dropout=dropout)
        return x4

pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)


bert = BertModel.from_pretrained(pretrained_weights,
                                 output_hidden_states=True,
                                 output_attentions=True, force_download=True)


class Bert(nn.Module):
    def __init__(self):
        # Assign instance variables
        super(Bert, self).__init__()
        self.bert = copy.deepcopy(bert)

    def forward(self, x, attention_mask, token_type_ids):

        outputs= self.bert(x, attention_mask=attention_mask, token_type_ids=token_type_ids)
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
        self.linear = nn.Linear(hidden, hashtag_size)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        return self.linear(x[:, 0])




class BERTLM_new(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = Bert()
        self.next_sentence = NextSentencePrediction(768)

        self.Loss = 0
        self.softmax = nn.Softmax(-1)
        self.transformer_new=Transformer_new()

    def forward(self, batch_loc_text, batch_img, batch_know, segment, dropout):
        attention_mask=(batch_loc_text!=tokenizer.pad_token_id)
        x = self.bert(batch_loc_text, attention_mask=attention_mask.float(), token_type_ids=segment)
        x=self.transformer_new(x, attention_mask, dropout)
        return self.next_sentence(x)

    def bptt(self, batch_loc_text, batch_img, batch_know, segment, label, dropout):  # (batch_size, out_len)
        next_sentence = self.forward(batch_loc_text, batch_img, batch_know, segment, dropout)
        sigmoid = torch.sigmoid(next_sentence)

        re_label = (label == 0).float()
        p = (re_label - sigmoid) * (re_label - label)
        # focal_factor=(1-p)**2
        loss_matrix = -torch.log(p)  # *focal_factor
        loss = torch.mean(loss_matrix)
        """""
        softmax=self.softmax(next_sentence)
        p=softmax[label==1]
        print(p)
        pos_matrix=-torch.log(p)

        loss=torch.mean(pos_matrix)
        """""
        return loss

        # Performs one step of SGD.

    def numpy_sdg_step(self, batch, optimizer, lrate, dropout):
        # Calculate the gradients

        optimizer.zero_grad()
        loss = self.bptt(torch.tensor(batch[0]).to(device).long(), torch.tensor(batch[1]).to(device).float(),
                         torch.tensor(batch[2]).to(device).long(), torch.tensor(batch[3]).to(device).long(),
                         torch.tensor(batch[4]).to(device).float(), dropout)

        loss.backward()
        optimizer.step()
        # optimizer.param_groups[0]['lr'] = lrate

        return loss

    def train_with_batch(self, info, batch_val, optimizer):
        val_precision = []

        Loss_len = 0
        num_examples_seen = 1
        # nepoch=nb_epoch()
        nepoch = 1000
        print('training epoch :', nepoch)
        print('lengh of batch :', int(len(info[0]) / SEN_LEN))
        for epoch in range(nepoch):
            epoch=epoch
            print("에폭", epoch + 1)
            batch_train = generate_batch(info[0], info[1], info[2], info[3])
            for i in range(len(batch_train)):

                batch_loc_text, batch_img, batch_know, segment, batch_y, true_label = batch_train[i]
                # print(np.shape(batch_know)) # [batch_size, cat_num]
                # 3factors or 4factors
                lrate = math.pow(64, -0.5) * min(math.pow(num_examples_seen, -0.5), num_examples_seen * math.pow(4000,
                                                                                                                 -1.5))  # warm up step : default 4000
                loss = self.numpy_sdg_step((batch_loc_text, batch_img, batch_know, segment, batch_y), optimizer,
                                           lrate, True)
                self.Loss += loss.item()
                Loss_len += 1

                if num_examples_seen % 1000 == 0:  # origin = int(batch_len * nepoch / 100)
                    time_ = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(time_, ' ', int(100 * num_examples_seen / (len(batch_train) * nepoch)), end='')
                    print('%   완료!!!', end='')
                    print('   loss :', self.Loss / Loss_len)  # , '   lr :', lrate)
                    self.Loss = 0
                    Loss_len = 0

                num_examples_seen += 1

            print('Epoch', epoch + 1, 'completed out of', nepoch)
            # valid set : 8104개
            val_pred = []
            val_truth = []

            for i in range(len(batch_val)):
                model.eval()
                batch_loc_text, batch_img, batch_know, segment, batch_y, true_label = batch_val[i]
                val_prob = self.softmax(
                    self.forward(torch.tensor(batch_loc_text).to(device).long(), torch.tensor(batch_img).to(device).float(),
                                 torch.tensor(batch_know).to(device).long(), torch.tensor(segment).to(device).long(), False))

                # if last_function() == "softmax":
                y_pred = np.argsort(val_prob.detach().cpu().numpy(), axis=1)
                for i in range(y_pred.shape[0]):
                    val_pred.append(y_pred[i])
                    val_truth.append(true_label[i])
            precision = top1_acc(val_truth, val_pred)

            print("Epoch:", (epoch + 1), "val_precision:", precision)
            val_precision.append(precision)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'num_examples_seen' : num_examples_seen
            }, 'project3_bert24_checkpoint/epoch='+str(epoch))

        return val_precision




def max_length(list):
    max = 0
    for i in list:
        if max < len(i):
            max = len(i)

    return max


SEN_LEN = 5


# Train on a small subset of the data to see what happens
def generate_batch(new_loc_text_list, new_loc_list, data, len_hashtag):
    batch = []

    temp_loc_text = []
    temp_seg=[]
    temp_img = []
    temp_know = []
    temp_y = []
    temp_y_ = []

    random.shuffle(new_loc_text_list)
    key = new_loc_text_list


    k = max_length(key[:SEN_LEN]) - 1
    if k > 512:
        k = 512
    max_num_sen = SEN_LEN

    num_sen = 0
    total_num_sen = 0
    for i in key:
        loc_data=new_loc_list[i[0]][1:]
        loc_text_seg=[0]*len(loc_data)+[1]*(k-len(loc_data))
        temp_seg.append(loc_text_seg)
        loc_text_data=i[1:]
        if len(loc_text_data) <= k:
            temp_loc_text.append( loc_text_data+ [tokenizer.pad_token_id] * (k - len(loc_text_data)))
        else:
            # print('error')
            temp_loc_text.append(loc_text_data[:511] + [tokenizer.sep_token_id])

        temp_img.append(data[0][i[0]])
        temp_know.append(data[4][i[0]])

        one_hot_y = []

        for j in range(len_hashtag):
            if j in data[3][i[0]]:
                one_hot_y.append(1)
            else:
                one_hot_y.append(0)
        temp_y.append(one_hot_y)
        temp_y_.append(data[3][i[0]])

        num_sen += 1
        total_num_sen += 1
        if num_sen == max_num_sen:
            if total_num_sen >= len(key):
                break
            k = max_length(key[total_num_sen:total_num_sen + SEN_LEN]) - 1
            if k > 512:
                k = 512
            max_num_sen = SEN_LEN
            num_sen = 0

            batch.append((temp_loc_text, temp_img, temp_know, temp_seg, temp_y, temp_y_))
            temp_loc_text = []
            temp_seg = []
            temp_img = []
            temp_know = []
            temp_y = []
            temp_y_ = []

    batch.append((temp_loc_text, temp_img, temp_know, temp_seg, temp_y, temp_y_))

    return batch


if __name__ == '__main__':
    print("co-attention_" + evaluation_factor())
    if evaluation_factor() == '1factor':
        print("factor:", which_factor())
    if evaluation_factor() == '2factors' or evaluation_factor() == '3factors':
        print("connection:", connection())

    print("last function:", last_function())
    print("current working directory:", os.getcwd())
    print('loading data...')

    with open("project3_data/vocabulary_keras_h.pkl", "rb") as f:
        data = pickle.load(f)
    vocabulary = data[0]
    hashtagVoc = data[2]
    vocabulary_inv = {}
    hashtagVoc_inv = {}
    hashtagCount = {}
    for k, v in vocabulary.items():
        vocabulary[k] = v + 2

    vocabulary["<Padding>"] = 0
    vocabulary['<CLS>'] = 1
    vocabulary['<SEP>'] = 2

    for i in vocabulary.keys():
        vocabulary_inv[vocabulary[i]] = i
    for i in hashtagVoc.keys():
        hashtagVoc_inv[hashtagVoc[i]] = i
        hashtagCount[hashtagVoc[i]] = []

    print("vocabulary 스펙:", len(vocabulary), max(vocabulary.values()), min(vocabulary.values()))
    print("hashtagVoc 스펙 :", len(hashtagVoc), max(hashtagVoc.values()), min(hashtagVoc.values()))
    print("len(hashtagVoc_inv)", len(hashtagVoc_inv))

    # Knowledge-base 추가
    k_train, k_val, k_test = load_k()
    print(len(k_train), len(k_val), len(k_test))
    ################################# for padding, cls, sep ########################################################
    for index, categories in enumerate(k_train):
        temp_category = []
        for category in categories:
            temp_category.append(category + 2)
        temp_category.append(2)
        k_train[index] = temp_category

    for index, categories in enumerate(k_val):
        temp_category = []
        for category in categories:
            temp_category.append(category + 2)

        temp_category.append(2)
        k_val[index] = temp_category

    for index, categories in enumerate(k_test):
        temp_category = []
        for category in categories:
            temp_category.append(category + 2)

        temp_category.append(2)
        k_test[index] = temp_category

    ####################################################################################################################

    val_data = []
    val_data.append(np.load("project3_transformer_data/transformer_image_90_val.npy"))
    print("validation data loading finished.")
    with open("project3_data/val_tlh_keras_h.bin", "rb") as f:
        val_data.extend(pickle.load(f))
    print("validation data 업로드")

    new_val_text_list = []
    for index, sentece in enumerate(val_data[1]):
        temp_sen_ = []
        for word in sentece:
            if word == 0:
                pass
            else:
                temp_sen_.append(word + 2)
        temp_sen_ = [tokenizer.sep_token_id]+tokenizer.encode(' '.join([vocabulary_inv[j] for j in temp_sen_]))+[tokenizer.sep_token_id]
        new_val_text_list.append(temp_sen_)

    new_val_loc_list = []
    for index, sentence in enumerate(val_data[2][:, -17:]):
        temp_sen = []
        for word in sentence:
            if word == 0:
                pass
            else:
                temp_sen.append(word + 2)
        temp_sen=[tokenizer.cls_token_id]+tokenizer.encode(' '.join([vocabulary_inv[j] for j in temp_sen]))+[tokenizer.sep_token_id]
        temp_sen.insert(0, index)
        new_val_loc_list.append(temp_sen)

    new_val_loc_text_list=[]

    for x, y in zip(new_val_loc_list, new_val_text_list):
        new_val_loc_text_list.append(x+y[1:])

    val_data.append(k_val)

    print(len(val_data[0]), len(val_data[1]), len(val_data[2]), len(val_data[3]), len(val_data[4]))
    # val_data = check_hashzero(val_data)
    # print("check 완")

    test_data = []
    test_data.append(np.load("project3_transformer_data/transformer_image_90_test.npy"))
    print("test data loading finished.")
    with open("project3_data/test_tlh_keras_h.bin", "rb") as f:
        test_data.extend(pickle.load(f))
    print("test data 업로드")

    new_test_text_list = []
    for index, sentece in enumerate(test_data[1]):
        temp_sen_ = []
        for word in sentece:
            if word == 0:
                pass
            else:
                temp_sen_.append(word + 2)
        temp_sen_ = [tokenizer.sep_token_id]+tokenizer.encode(' '.join([vocabulary_inv[j] for j in temp_sen_]))+[tokenizer.sep_token_id]
        new_test_text_list.append(temp_sen_)

    new_test_loc_list = []
    for index, sentece in enumerate(test_data[2][:, -17:]):
        temp_sen = []
        for word in sentece:
            if word == 0:
                pass
            else:
                temp_sen.append(word + 2)
        temp_sen=[tokenizer.cls_token_id]+tokenizer.encode(' '.join([vocabulary_inv[j] for j in temp_sen]))+[tokenizer.sep_token_id]
        temp_sen.insert(0, index)
        new_test_loc_list.append(temp_sen)

    new_test_loc_text_list = []

    for x, y in zip(new_test_loc_list, new_test_text_list):
        new_test_loc_text_list.append(x+y[1:])


    test_data.append(k_test)

    print(len(test_data[0]), len(test_data[1]), len(test_data[2]), len(test_data[3]), len(test_data[4]))
    # test_data = check_hashzero(test_data)
    # print("check 완")

    train_data = []
    train_data.append(np.load("project3_transformer_data/transformer_image_90_train.npy"))
    print("train data loading finished.")
    # with open("./project3_data/train_tlh_keras_h.bin", "rb") as f:
    with open("project3_data/train_tlh_keras_h.bin", "rb") as f:
        train_data.extend(pickle.load(f))
    print("train data 업로드")

    ############################################################################
    batch_val = generate_batch(new_val_loc_text_list, new_val_loc_list, val_data, len(hashtagVoc))
    batch_test = generate_batch(new_test_loc_text_list, new_test_loc_list, test_data, len(hashtagVoc))
    ############################################################################

    new_train_text_list = []
    for index, sentece in enumerate(train_data[1]):
        temp_sen_ = []
        for word in sentece:
            if word == 0:
                pass
            else:
                temp_sen_.append(word + 2)
        temp_sen_ = [tokenizer.sep_token_id]+tokenizer.encode(' '.join([vocabulary_inv[j] for j in temp_sen_]))+[tokenizer.sep_token_id]
        new_train_text_list.append(temp_sen_)

    new_train_loc_list = []
    for index, sentence in enumerate(train_data[2][:, -17:]):
        temp_sen = []
        for word in sentence:
            if word == 0:
                pass
            else:
                temp_sen.append(word + 2)
        temp_sen = [tokenizer.cls_token_id]+tokenizer.encode(' '.join([vocabulary_inv[j] for j in temp_sen]))+[tokenizer.sep_token_id]
        temp_sen.insert(0, index)
        new_train_loc_list.append(temp_sen)

    new_train_loc_text_list = []

    for x, y in zip(new_train_loc_list, new_train_text_list):
        new_train_loc_text_list.append(x+y[1:])

    train_data.append(k_train)

    print(len(train_data[0]), len(train_data[1]), len(train_data[2]), len(train_data[3]))
    print("train data size: ", len(train_data[3]))

    text_maxlen = len(train_data[1][0])  # 411 맞는지 확인
    loc_maxlen = len(train_data[2][0])  # 19
    print("text_maxlen:", text_maxlen)
    print("loc_maxlen:", loc_maxlen)
    vocab_size = len(vocabulary_inv)  # 26210
    hashtag_size = len(hashtagVoc_inv)  # 2988
    print('-')
    print('Vocab size:', vocab_size, 'unique words')
    print('Hashtag size:', hashtag_size, 'unique hashtag')

    for index, taglist in enumerate(train_data[3]):
        for tag in taglist:
            hashtagCount[int(tag)].append(index)

    cnt = 0
    for i in list(hashtagCount.keys()):
        if len(hashtagCount[i]) == 0:
            del hashtagCount[i]

    hashtagCount_saved = copy.deepcopy(hashtagCount)

    torch.manual_seed(10)

    model = BERTLM_new(vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), eps=1e-9, lr=1e-5)
    print("starts training...")

    val_precision = model.train_with_batch((new_train_loc_text_list, new_train_loc_list, train_data, len(hashtagVoc)), batch_val, optimizer)

    print("\ntop1 결과 정리")
    print("validation")
    for i in range(len(val_precision)):
        print("epoch", i + 1, "- precision:", val_precision[i])
    # print("attention vector weights")
    # print_order = ["i_w_it", "i_w_il", "i_w_ik", "t_w_it", "t_w_lt", "t_w_tk", "l_w_lt",
    #                "l_w_il", "l_w_lk", "k_w_ik", "k_w_tk", "k_w_lk"]
    # print(attention_weights)
    # with open("./result_weight/" + evaluation_factor() + ".txt", "w") as ff:
    #     ff.write("순서대로 i_w_it, i_w_il, i_w_ik, t_w_it, t_w_lt, t_w_tk, l_w_lt, l_w_il, l_w_lk, k_w_ik, k_w_tk, k_w_lk")
    #     for q in range(len(attention_weights)):
    #         ff.write(print_order[q] + " : " + str(attention_weights[q]) + "\n")
    # with open("./result_weight/" + evaluation_factor() + ".bin", "wb") as ff:
    #     pickle.dump(attention_weights, ff)
    #