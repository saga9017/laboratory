import pickle
import os

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
import torch.nn.utils as torch_utils


from transformers import *

# pretrained_weights = 'bert-base-uncased'
# tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
#
#
# with open('textnextract.pickle', 'rb') as f:
#     multi_modal_dic = pickle.load(f)
#
# dic_list = list(multi_modal_dic)
#
# new_train_text_list = []
# train_label = []
# for key in dic_list[10:100]:
#     text, label, extract = multi_modal_dic[key]
#     tmp_sen = tokenizer.encode(extract) + tokenizer.encode(text)[1:]
#     new_train_text_list.append(tmp_sen)
#     train_label.append(label)
#
# # print(new_train_text_list)
#
#
# def generate_batch(corpus, label):
#     corpus_len = len(corpus)
#     iters = int(corpus_len / 10)
#     tmp_batch = []
#     tmp_batch_label = []
#     for iter in range(iters):
#         tmp = corpus[iter*10:(iter+1)*10]
#         tmp2 = []
#         max_len = 0
#         for sentence in tmp:
#             if max_len < len(sentence):
#                 max_len = len(sentence)
#         for sentence in tmp:
#             sentence = sentence + [0] *(max_len -len(sentence))
#             tmp2.append(sentence)
#         tmp_batch.append(tmp2)
#
#         tmp_batch_label.append(label[iter*10:(iter+1)*10])
#     tmp_batch_label.append(label[iters*10:])
#     return tmp_batch, tmp_batch_label
#
# tmp_batch, tmp_label = generate_batch(new_train_text_list, train_label)
#
# print(tmp_label[0][0])
