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
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # GPU number to use

print('gpu :', torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_default_dtype(torch.float)

print('cudnn :', torch.backends.cudnn.enabled == True)
torch.backends.cudnn.benchmark=True
#####################################################
view_sentence_len = -1
unknown_number = 1
MAX_LENGTH = 160
MAX_TOKEN=10000
#ITERATION=200*1000
hidden_dim=512
Dropout=0.1
#####################################################



def Positional_Encoding(MAX_LENGTH, hidden_dim):  # x : (batch_size, input_len, hidden_dim) or (batch_size, output_len, hidden_dim)
    table = torch.zeros((MAX_LENGTH, hidden_dim))
    a, b = table.shape
    for pos in range(a):
        for i in range(b):
            if i % 2 == 0:
                table[pos][i] = math.sin(pos / math.pow(10000, i / b))
            else:
                table[pos][i] = math.cos(pos / math.pow(10000, (i - 1) / b))
    return Parameter(table, requires_grad=False)

table={}
for i in range(MAX_LENGTH+1):
    table[i+1]=Positional_Encoding(i+1, 300)

mask={}
for i in range(MAX_LENGTH+1):
    mask[i+1] = torch.triu(torch.ones((i+1, i+1)), diagonal=1).byte()

with open('w2v_lower.pkl', 'rb') as fin:
    embedded = pickle.load(fin)


embedded_matrix=torch.from_numpy(np.concatenate((np.random.randn(1,300), np.random.randn(1,300), embedded[0]), 0)).float().to(device)
w2i=embedded[1]
for k,v in w2i.items():
    w2i[k]=v+2

w2i['Null']=0
w2i['CLS']=1


def generate_batch(MAX_TOKEN):
    y_train = []
    X_train = []

    label_dic = {}
    re_label_dic = {}
    label_num = 0
    max_len = 1
    with open('cnn_data/odp.wsdm.train.all', 'r') as f:
        while True:
            line = f.readline().replace(',', '').split()
            if not line: break
            if int(line[0]) not in label_dic:
                label_dic[int(line[0])] = label_num
                re_label_dic[label_num] = int(line[0])
                label_num += 1

            y_train.append(label_dic[int(line[0])])
            temp = [len(X_train)]
            for index, i in enumerate(line[1:]):
                if index + 1 > max_len:
                    max_len = index + 1

                if i not in w2i.keys():
                    temp.append(w2i['unknown'])
                else:
                    temp.append(w2i[i])
            X_train.append(temp)

    y_number=max(y_train)+1

    temp_X = []
    temp_y = []
    batch = []

    key = list(reversed(sorted(X_train, key=len)))
    key_dic={}

    k = len(key[0][1:])

    max_num_sen = int(MAX_TOKEN / k)
    num_sen = 0

    for i in key:
        key_dic[len(key_dic)]=i[0]
        if len(i[1:]) <= k:
            temp_X.append([1]+i[1:] + [0] * (k - len(i[1:])))
        else:
            print('error')
            temp_X.append([1]+i[1:])

        temp_y.append(y_train[i[0]])

        num_sen += 1
        if num_sen == max_num_sen:
            k = len(key[num_sen][1:])
            max_num_sen = int(MAX_TOKEN / k)
            num_sen = 0

            batch.append((temp_X, table[len(temp_X[0])], temp_y))
            temp_X = []
            temp_y = []



    if len(temp_X)!=0:
        batch.append((temp_X, table[len(temp_X[0])], temp_y))

    y_dev = []
    X_dev = []
    with open('cnn_data/odp.wsdm.dev.all', 'r') as f:
        while True:
            line = f.readline().replace(',', '').split()
            if not line: break

            y_dev.append(label_dic[int(line[0])])
            temp = [len(X_dev)]
            for index, i in enumerate(line[1:]):
                if index + 1 > max_len:
                    continue

                if i not in w2i.keys():
                    temp.append(w2i['unknown'])
                else:
                    temp.append(w2i[i])
            X_dev.append(temp)

    temp_X = []
    temp_y = []
    batch_dev = []

    key = list(reversed(sorted(X_dev, key=len)))
    key_dic = {}

    k = len(key[0][1:])

    max_num_sen = int(MAX_TOKEN / k)
    num_sen = 0

    for i in key:
        key_dic[len(key_dic)] = i[0]
        if len(i[1:]) <= k:
            temp_X.append([1]+i[1:] + [0] * (k - len(i[1:])))
        else:
            print('error')
            temp_X.append([1]+i[1:])

        temp_y.append(y_dev[i[0]])

        num_sen += 1
        if num_sen == max_num_sen:
            k = len(key[num_sen][1:])
            max_num_sen = int(MAX_TOKEN / k)
            num_sen = 0

            batch_dev.append((temp_X, table[len(temp_X[0])], temp_y))
            temp_X = []
            temp_y = []

    if len(temp_X)!=0:
        batch_dev.append((temp_X, table[len(temp_X[0])], temp_y))

    y_test = []
    X_test = []
    with open('cnn_data/odp.wsdm.test.all', 'r') as f:
        while True:
            line = f.readline().replace(',', '').split()
            if not line: break

            y_test.append(label_dic[int(line[0])])
            temp = [len(X_test)]
            for index, i in enumerate(line[1:]):
                if index + 1 > max_len:
                    continue

                if i not in w2i.keys():
                    temp.append(w2i['unknown'])
                else:
                    temp.append(w2i[i])
            X_test.append(temp)

    temp_X = []
    temp_y = []
    batch_test = []

    key = list(reversed(sorted(X_test, key=len)))
    key_dic = {}

    k = len(key[0][1:])

    max_num_sen = int(MAX_TOKEN / k)
    num_sen = 0

    for i in key:
        key_dic[len(key_dic)] = i[0]
        if len(i[1:]) <= k:
            temp_X.append([1]+i[1:] + [0] * (k - len(i[1:])))
        else:
            print('error')
            temp_X.append([1]+i[1:])

        temp_y.append(y_test[i[0]])

        num_sen += 1
        if num_sen == max_num_sen:
            k = len(key[num_sen][1:])
            max_num_sen = int(MAX_TOKEN / k)
            num_sen = 0

            batch_test.append((temp_X, table[len(temp_X[0])], temp_y))
            temp_X = []
            temp_y = []

    if len(temp_X)!=0:
        batch_test.append((temp_X, table[len(temp_X[0])], temp_y))

    return batch, key_dic,  y_number, batch_dev, batch_test, label_dic

def onehot(index_array: np.ndarray, nb_classes: int, dtype=np.float32):
    return np.eye(nb_classes, dtype=dtype)[index_array]


def confusion_per_class(predictions: np.ndarray, labels: np.ndarray, nb_classes: int):
    predictions = onehot(predictions, nb_classes, np.bool8)
    labels = onehot(labels, nb_classes, np.bool8)

    n_predictions = ~predictions
    n_labels = ~labels

    tp_per_class = (predictions & labels).sum(0).astype(np.float32)
    fp_per_class = (predictions & n_labels).sum(0).astype(np.float32)
    fn_per_class = (n_predictions & labels).sum(0).astype(np.float32)
    return tp_per_class, fp_per_class, fn_per_class


def micro_f1_score(tp_per_class, fp_per_class, fn_per_class):
    total_tp = tp_per_class.sum()
    total_fp = fp_per_class.sum()
    total_fn = fn_per_class.sum()
    del tp_per_class
    del fp_per_class
    del fn_per_class

    total_precision = total_tp / (total_tp + total_fp + 1e-12)
    total_recall = total_tp / (total_tp + total_fn + 1e-12)

    micro_f1 = 2 * total_precision * total_recall / (total_precision + total_recall + 1e-12)
    del total_precision
    del total_recall

    return micro_f1


def www_macro_f1_score(tp_per_class: np.ndarray, fp_per_class: np.ndarray, fn_per_class: np.ndarray):
    is_nonzero_prediction = (tp_per_class + fp_per_class) != 0
    is_nonzero_actual = (tp_per_class + fn_per_class) != 0

    where_nonzero_prediction = np.squeeze(is_nonzero_prediction.nonzero(), axis=-1)
    where_nonzero_actual = np.squeeze(is_nonzero_actual.nonzero(), axis=-1)
    del is_nonzero_prediction, is_nonzero_actual

    precision_per_class = tp_per_class[where_nonzero_prediction] / (
                tp_per_class[where_nonzero_prediction] + fp_per_class[where_nonzero_prediction])
    recall_per_class = tp_per_class[where_nonzero_actual] / (tp_per_class[where_nonzero_actual] + fn_per_class[where_nonzero_actual])
    del tp_per_class, fp_per_class, fn_per_class

    macro_precision = precision_per_class.mean()
    macro_recall = recall_per_class.mean()
    del precision_per_class, recall_per_class

    if macro_precision + macro_recall==0:
        macro_f1=0
    else:
        macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall)

    return macro_f1


def compute_metrics(preds, labels, nb_classes):
    assert len(preds) == len(labels)

    tp_per_class, fp_per_class, fn_per_class = confusion_per_class(preds, labels, nb_classes)
    micro_f1 = micro_f1_score(tp_per_class, fp_per_class, fn_per_class)
    macro_f1 = www_macro_f1_score(tp_per_class, fp_per_class, fn_per_class)
    return {"micro_f1": micro_f1, "macro_f1": macro_f1}
#################################################### Module ############################################################

class Norm(nn.Module):
    def __init__(self):
        super(Norm, self).__init__()
        self.gamma=Parameter(torch.tensor(1.0))
        self.beta = Parameter(torch.tensor(0.0))
    def forward(self, X):
        mu = torch.mean(X, dim=-1, keepdim=True)
        var = torch.var(X, dim=-1, keepdim=True)
        X_norm = torch.div(X - mu, torch.sqrt(var + 1e-8))
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

        a = torch.cat((head3[0], head3[1], head3[2], head3[3], head3[4], head3[5], head3[6], head3[7]), 2)

        result = self.w_os(a)
        result=self.layerNorm_add_Norm(result+de)
        return result  # (output_len, hidden)

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
            output = self.dropout(F.relu(output))  # (batch_size, input_len, 4*hidden)
        else:
            output = F.relu(output)
        output = self.fc2(output)  # (batch_size, input_len, hidden
        output=self.layerNorm_add_Norm(output+x)
        return output

##################################################### Sub layer ########################################################

class Encoder_layer(nn.Module):
    def __init__(self, hidden_dim, hidden_dim_, dropout=Dropout):  # default=512
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
class Encoder(nn.Module):
    def __init__(self ,hidden_dim, hidden_dim_):  # default=512
        # Assign instance variables
        super().__init__()
        self.hidden = hidden_dim
        self.embedded_matrix=Parameter(embedded_matrix)
        self.encoder1=Encoder_layer(hidden_dim, hidden_dim_)
        self.encoder2 = Encoder_layer(hidden_dim, hidden_dim_)
        self.encoder3 = Encoder_layer(hidden_dim, hidden_dim_)
        self.encoder4 = Encoder_layer(hidden_dim, hidden_dim_)
        self.encoder5 = Encoder_layer(hidden_dim, hidden_dim_)
        self.encoder6 = Encoder_layer(hidden_dim, hidden_dim_)

    def input_embedding(self, x):  # x: (batch, input_len, )
        mask_e=Parameter((x==0).unsqueeze(1).repeat(1,x.shape[1], 1), requires_grad=False)
        mask_d = Parameter((x == 0), requires_grad=False)
        return self.embedded_matrix[x], mask_e, mask_d  # (input_len, hidden_dim)

    def forward(self, x, table_x, dropout):
        non_pad_mask = (x != 0).unsqueeze(-1).repeat(1, 1, self.hidden).float()
        x1, mask_e, mask_d = self.input_embedding(x)  # (input_len, hidden)
        x1=x1.masked_fill(non_pad_mask==0, 0)
        x2 = (x1 + table_x).masked_fill(non_pad_mask==0, 0)  # (input_len, hidden)
        ########################################################
        x3 = self.encoder1(x2 + x1, mask_e, non_pad_mask, dropout=dropout)
        x4 = self.encoder2(x3 + x2 + x1, mask_e, non_pad_mask, dropout=dropout)
        x5 = self.encoder3(x4 + x3 + x2 + x1, mask_e, non_pad_mask, dropout=dropout)
        x6 = self.encoder4(x5 + x4 + x3 + x2 + x1, mask_e, non_pad_mask, dropout=dropout)
        x7 = self.encoder5(x6 + x5 + x4 + x3 + x2 + x1, mask_e, non_pad_mask, dropout=dropout)
        x8 = self.encoder6(x7 + x6 + x5 + x4 + x3 + x2 + x1, mask_e, non_pad_mask, dropout=dropout)

        return x8, mask_d




class transformer(nn.Module):
    def __init__(self, word_dim2,  hidden_dim=300, hidden_dim_=hidden_dim, label_smoothing=0.1):  # default=512
        # Assign instance variables
        super(transformer, self).__init__()
        self.hidden = hidden_dim
        self.softmax=nn.Softmax(-1)
        self.Loss = 0

        self.encoder=Encoder(hidden_dim, hidden_dim_)

        self.V_d = nn.Linear(hidden_dim, word_dim2)
        nn.init.xavier_normal_(self.V_d.weight)

        self.target_prob = Parameter(torch.tensor((1 - label_smoothing) + label_smoothing / word_dim2),
                                     requires_grad=False)
        self.nontarget_prob = Parameter(torch.tensor(label_smoothing / word_dim2), requires_grad=False)

    def forward_propagation(self, x, table_x):
        x, mask_d=self.encoder(x, table_x, dropout=True)
        x = self.softmax(self.V_d(x[:,0]))
        return x


    def bptt(self, x, table_x, y ):  # (batch_size, out_len)
        x = self.forward_propagation(x, table_x)
        pos = torch.log(torch.gather(x, 1, y.view(-1, 1)))
        neg = torch.sum(torch.log(x), dim=1) - pos
        loss = -self.target_prob * pos - self.nontarget_prob * neg
        loss = torch.mean(loss)
        return loss


    def predict(self, x, table_):
        x=torch.tensor(x).to(device)
        table_=table_.to(device)
        ###############################################################################################################
        x, mask_d=self.encoder(x, table_, dropout=False)
        ###############################################################################################################
        x = self.softmax(self.V_d(x[:,0]))
        output = torch.argmax(x, 1)
        return output

    def epoch_eval(self, batch_dev):

        score=0.0
        micro=0.0
        macro=0.0
        num_flags=0
        seen=0
        for i in range(len(batch_dev)):
            prediction=self.predict(torch.tensor(batch_dev[i][0]).to(device),torch.tensor(batch_dev[i][1]).to(device)).cpu()
            flags = ( prediction== torch.tensor(batch_dev[i][2]))
            num_flags+=len(flags)
            score += torch.sum(flags).item()
            f1_score=compute_metrics(prediction, torch.tensor(batch_dev[i][2]), 2531)
            micro+=f1_score['micro_f1']
            macro+=f1_score['macro_f1']
            seen+=1

        return score/num_flags, micro/seen, macro/seen


    # Performs one step of SGD.
    def numpy_sdg_step(self, batch, optimizer):
        # Calculate the gradients

        optimizer.zero_grad()
        loss = self.bptt(torch.tensor(batch[0]).to(device), batch[1].to(device), torch.tensor(batch[2]).to(device))
        loss.backward()
        optimizer.step()

        return loss

    def train_with_batch(self, batch, batch_dev, batch_test, evaluate_loss_after=5):
        # We keep track of the losses so we can plot them later
        num_examples_seen = 1
        Loss_len = 0
        last_loss=0
        #nepoch=int(ITERATION/len(batch))
        nepoch=100
        print('epoch :', nepoch)
        optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=0.00005)
        for epoch in range(nepoch):
            # For each training example...
            for i in range(len(batch)):
                # One SGD step
                self.Loss += self.numpy_sdg_step(batch[i], optimizer).item()

                Loss_len += 1
                if num_examples_seen == len(batch)*nepoch:
                    last_loss= self.Loss / Loss_len
                else:
                    if int(len(batch) * nepoch/100)==0:
                        time_ = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        print(time_, ' ', int(100 * num_examples_seen / (len(batch) * nepoch)), end='')
                        print('%   완료!!!', end='')
                        print('   loss :', self.Loss / Loss_len)
                        self.Loss = 0
                        Loss_len = 0
                    else:
                        if num_examples_seen %  300== 0:
                            time_ = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            print(time_, ' ', int(100 * num_examples_seen / (len(batch) * nepoch)), end='')
                            print('%   완료!!!', end='')
                            print('   loss :', self.Loss / Loss_len)
                            self.Loss = 0
                            Loss_len = 0
                num_examples_seen += 1
            score=self.epoch_eval(batch_dev)
            score_test = self.epoch_eval(batch_test)
            print('=================epoch :', epoch+1, "=====================")
            print('dev accuracy :', score[0])
            print('dev micro f1 score :', score[1])
            print('dev macro f1 score :', score[2])
            print()
            print('test accuracy :', score_test[0])
            print('test micro f1 score :', score_test[1])
            print('test macro f1 score :', score_test[2])
            torch.save(model.state_dict(), 'transformer_final2_saved_model/checkpoint'+str(epoch))
            print('saved')
        return last_loss

torch.manual_seed(10)
batch, key_dic, y_number, batch_dev, batch_test, label_dic= generate_batch(MAX_TOKEN)
torch.save(label_dic, 'transformer_final2_label_dic')
torch.save(key_dic, 'transformer_final2_key_dic')
torch.save(y_number, 'transformer_final2_y_number')
# Train on a small subset of the data to see what happens
model = transformer(y_number).to(device)
print('preprocess done!!!')


last_loss = model.train_with_batch(batch, batch_dev, batch_test)







