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
MAX_LENGTH = 100
MAX_TOKEN=1500
ITERATION=10*1000
hidden_dim=384
Dropout=0.1
#####################################################

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

SST_2_train = []
SST_2_dev = []
SST_2_test = []
num_classes = 2  # the number of classes

# sentences of training data
with open('SST-2/stsa.binary.train', 'r', encoding='latin-1') as f:
    for row in f.readlines():
        SST_2_train.append([clean_str_sst(row[1:]), int(row[0])])

# sentences of validation data
with open('SST-2/stsa.binary.dev', 'r', encoding='latin-1') as f:
    for row in f.readlines():
        SST_2_dev.append([clean_str_sst(row[1:]), int(row[0])])

# sentences of test data
with open('SST-2/stsa.binary.test', 'r', encoding='latin-1') as f:
    for row in f.readlines():
        SST_2_test.append([clean_str_sst(row[1:]), int(row[0])])


print(SST_2_test)
w2i=torch.load('w2i_bert_sample')

X_train=[]
X_train_seg=[]
y_train=[]
for sen in SST_2_train:
    y_train.append(sen[1])
    temp=[1]
    temp_seg=[1,1]
    for word in sen[0].split():
        temp_seg.append(1)
        if word not in w2i:
            temp.append(3)
        else:
            temp.append(w2i[word])
    temp.append(2)
    X_train.append(temp)
    X_train_seg.append(temp_seg)

print('X_train :', len(X_train))
print('X_train_seg :', len(X_train_seg))
print('y_train :', len(y_train))

X_test=[]
X_test_seg=[]
y_test=[]
for sen in SST_2_test:
    y_test.append(sen[1])
    temp=[1]
    temp_seg = [1, 1]
    for word in sen[0].split():
        temp_seg.append(1)
        if word not in w2i:
            temp.append(3)
        else:
            temp.append(w2i[word])
    temp.append(2)
    X_test.append(temp)
    X_test_seg.append(temp_seg)


print('X_test :', X_test)
print('X_test_seg :', X_test_seg)
print('y_test :', y_test)

batch_train=[]

temp_X=[]
temp_X_seg=[]
temp_y=[]
for i in range(len(X_train)):
    if len(X_train[i])<=100:
        temp_X.append(X_train[i]+[0]*(100-len(X_train[i])))
        temp_X_seg.append(X_train_seg[i]+[0]*(100-len(X_train[i])))
        temp_y.append(y_train[i])
    else:
        print('error')
    if len(temp_X)==MAX_TOKEN/100:
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
        temp_X.append(X_test[i] + [0] * (100 - len(X_test[i])))
        temp_X_seg.append(X_test_seg[i] + [0] * (100 - len(X_test[i])))
        temp_y.append(y_test[i])
    else:
        print('error')
    if len(temp_X) == MAX_TOKEN / 100:
        batch_test.append((temp_X, temp_X_seg, temp_y))
        temp_X = []
        temp_X_seg = []
        temp_y = []

if temp_X!=[]:
    batch_test.append((temp_X, temp_X_seg, temp_y))



def get_pad_mask(x):
    z=torch.zeros(x.shape[0], x.shape[1], hidden_dim)
    result=x.nonzero()

    return result
#################################################### Module ############################################################

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


        q = self.w_qs(de).view(len_d0, len_d1, -1, 12).permute(3, 0, 1, 2)
        k = self.w_ks(en).view(len_e0, len_e1, -1, 12).permute(3, 0, 2, 1)
        v = self.w_vs(en).view(len_e0, len_e1, -1, 12).permute(3, 0, 1, 2)

        e = torch.matmul(q, k) / math.sqrt(d_k)
        masked_e = e.masked_fill(mask, -1e10)
        alpha = self.softmax(masked_e)  # (output_len, input_len)
        if dropout==True:
            alpha = self.dropout(alpha)
        head3 = torch.matmul(alpha, v)

        a = torch.cat((head3[0], head3[1], head3[2], head3[3], head3[4],
                       head3[5], head3[6], head3[7], head3[8], head3[9],
                       head3[10], head3[11]), 2)

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
class Bert(nn.Module):
    def __init__(self, word_dim1, hidden_dim=hidden_dim, hidden_dim_=hidden_dim):  # default=512
        # Assign instance variables
        super().__init__()
        self.hidden = hidden_dim
        self.token_embed = nn.Embedding(word_dim1, hidden_dim)
        self.segment_embed = nn.Embedding(3, hidden_dim)
        self.sequence_embed = Parameter(torch.randn(MAX_LENGTH, hidden_dim))
        self.token_embed.weight.data.uniform_(-0.02, 0.02)
        self.segment_embed.weight.data.uniform_(-0.02, 0.02)
        nn.init.xavier_normal_(self.sequence_embed)

        self.encoder1=Encoder_layer(hidden_dim, hidden_dim_)
        self.encoder2 = Encoder_layer(hidden_dim, hidden_dim_)
        self.encoder3 = Encoder_layer(hidden_dim, hidden_dim_)
        self.encoder4 = Encoder_layer(hidden_dim, hidden_dim_)
        self.encoder5 = Encoder_layer(hidden_dim, hidden_dim_)
        self.encoder6 = Encoder_layer(hidden_dim, hidden_dim_)

    def input_embedding(self, x, seg):  # x: (batch, input_len, )
        mask_e=Parameter((x==0).unsqueeze(1).repeat(1,x.shape[1], 1), requires_grad=False)
        return self.token_embed(x)+self.segment_embed(seg)+self.sequence_embed[:x.shape[1]].unsqueeze(0).repeat(x.shape[0], 1, 1),  mask_e  # (input_len, hidden_dim)

    def forward(self, x, seg, dropout):
        non_pad_mask = (x != 0).unsqueeze(-1).repeat(1, 1, self.hidden).float()
        x1, mask_e= self.input_embedding(x, seg)  # (input_len, hidden)
        x1=x1.masked_fill(non_pad_mask==0, 0)
        ########################################################
        x2 = self.encoder1(x1, mask_e, non_pad_mask, dropout=dropout)
        x3 = self.encoder2(x2 + x1, mask_e, non_pad_mask, dropout=dropout)
        x4 = self.encoder3(x3 + x2 + x1, mask_e, non_pad_mask, dropout=dropout)
        x5 = self.encoder4(x4 + x3 + x2 + x1, mask_e, non_pad_mask, dropout=dropout)
        x6 = self.encoder5(x5 + x4 + x3 + x2 + x1, mask_e, non_pad_mask, dropout=dropout)
        x7 = self.encoder6(x6 + x5 + x4 + x3 + x2 + x1, mask_e, non_pad_mask, dropout=dropout)
        return x7

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
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


class BERTLM(nn.Module):
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
        self.bert = Bert(vocab_size, hidden_dim=hidden_dim, hidden_dim_=hidden_dim)
        self.next_sentence = NextSentencePrediction(self.bert.hidden)
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)
        self.Loss=0
        self.criterion_next_sen=nn.NLLLoss()

    def forward(self, x, segment_label, dropout):
        x = self.bert(x, segment_label, dropout)
        return self.next_sentence(x), self.mask_lm(x)

    def bptt(self, x, segment, y):  # (batch_size, out_len)
        next_sentence, mask_lm = self.forward(x, segment, True)

        # 2-1. NLL(negative log likelihood) loss of is_next classification result
        next_loss = self.criterion_next_sen(next_sentence, y.squeeze(-1))
        # 2-2. NLLLoss of predicting masked token word
        # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
        loss = next_loss

        return loss

    def evaluation(self, batch_test):
        score=0
        sample_num=0
        for i in range(len(batch_test)):
            next_sentence, mask_lm = self.forward(torch.tensor(batch_test[i][0]).to(device), torch.tensor(batch_test[i][1]).to(device), False)
            matching=(torch.argmax(next_sentence, dim=1)==torch.tensor(batch_test[i][2]).to(device))
            score+=torch.sum(matching).float()
            sample_num+=len(matching)

        print('score :', score/sample_num)

        # Performs one step of SGD.
    def numpy_sdg_step(self, batch, optimizer):
        # Calculate the gradients

        optimizer.zero_grad()
        loss = self.bptt(torch.tensor(batch[0]).to(device), torch.tensor(batch[1]).to(device),
                         torch.tensor(batch[2]).to(device))
        loss.backward()
        optimizer.step()

        return loss

    def train_with_batch(self, batch, batch_test, evaluate_loss_after=5):
        # We keep track of the losses so we can plot them later
        num_examples_seen = 1
        Loss_len = 0
        last_loss = 0
        nepoch = int(ITERATION / len(batch))+1
        print('epoch :', nepoch)
        optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), eps=1e-9, lr=0.00002)
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

model = BERTLM(len(w2i)).to(device)
model.load_state_dict(torch.load('saved_bert_sample'))
print('preprocess done!!!')
last_loss = model.train_with_batch(batch_train, batch_test)

print(last_loss)




"""""
predicts=[]
for i in range(len(batch)):
    predicts.append(model.predict(batch[i][0], batch[i][1]))

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
            f.write('predict : %s\n' % [output_lang.index2word[i.item()] for i in k[j] if i != 0])
            f.write('\n')
        pre_k+=len(k)
    f.close()
    print("저장 완료!!!")



save('[final]view_sentence='+str(view_sentence_len)+' batch_size='+str(MAX_TOKEN)+' iteration='+str(ITERATION)+'.txt')

torch.save(model.state_dict(), 'saved_model')
"""""