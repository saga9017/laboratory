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
#view_sentence_len = 1024
#unknown_number = 5
MAX_LENGTH = 100
MAX_TOKEN=2000
ITERATION=10*1000
hidden_dim=384
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

    def forward(self, x, segment_label):
        x = self.bert(x, segment_label, 0.1)
        return self.next_sentence(x), self.mask_lm(x)

    def bptt(self, x, y, segment, next_sen_label, masked_index):  # (batch_size, out_len)
        next_sentence, mask_lm = self.forward(x, segment)

        # 2-1. NLL(negative log likelihood) loss of is_next classification result
        next_loss = self.criterion_next_sen(next_sentence, next_sen_label.squeeze(-1))
        # 2-2. NLLLoss of predicting masked token word
        mask_loss = -torch.gather(mask_lm, 2, y.unsqueeze(-1)).squeeze(-1)*masked_index.float()
        mask_loss=torch.sum(mask_loss)/torch.sum(masked_index.float())
        # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
        loss = next_loss + mask_loss

        """""
        a, b = y_.nonzero().t()[0], y_.nonzero().t()[1]
        z = y9[a, b]
        pos = torch.log(z.gather(1, y_[a, b].unsqueeze(-1))).squeeze()
        neg = torch.sum(torch.log(z), dim=1) - pos
        loss = -self.target_prob * pos - self.nontarget_prob * neg
        loss = torch.mean(loss)
        """

        return loss

        # Performs one step of SGD.
    def numpy_sdg_step(self, batch, optimizer):
        # Calculate the gradients

        optimizer.zero_grad()
        loss = self.bptt(torch.tensor(batch[0]).to(device), torch.tensor(batch[1]).to(device),
                         torch.tensor(batch[2]).to(device), torch.tensor(batch[3]).to(device),
                         torch.tensor(batch[4]).to(device) )
        loss.backward()
        optimizer.step()

        return loss

    def train_with_batch(self, batch, evaluate_loss_after=5):
        # We keep track of the losses so we can plot them later
        num_examples_seen = 1
        Loss_len = 0
        last_loss = 0
        nepoch = int(ITERATION / len(batch))
        print(len(batch))
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
        return last_loss



torch.manual_seed(10)
# Train on a small subset of the data to see what happens

model = BERTLM(len(w2i)).to(device)

print('preprocess done!!!')
last_loss = model.train_with_batch(batch)

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



save('[bert]view_sentence='+str(view_sentence_len)+' batch_size='+str(MAX_TOKEN)+' iteration='+str(ITERATION)+'.txt')
"""
torch.save(model.state_dict(), 'saved_bert')
torch.save(w2i, 'w2i_bert')
