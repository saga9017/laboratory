import pickle
import numpy as np
import torch

torch.set_default_dtype(torch.double)

with open('/content/drive/My Drive/w2v_lower.pkl', 'rb') as fin:
    embedded = pickle.load(fin)


embedded_matrix=torch.from_numpy(np.concatenate((embedded[0], np.zeros((1,300))), 0)).cuda()
w2i=embedded[1]

##############단어하나당 embedding 300####################
#print(w2i.keys())

y_train=[]
X_train=[]

max_len=1

label_dic={}
re_label_dic={}
label_num=0
with open('/content/drive/My Drive/cnn_data/odp.wsdm.train.all', 'r') as f:
    while True:
        line = f.readline().replace(',', '').split()
        if not line: break
        if int(line[0]) not in label_dic:
            label_dic[int(line[0])]=label_num
            re_label_dic[label_num]=int(line[0])
            label_num+=1

        y_train.append(label_dic[int(line[0])])
        temp=[]
        for index, i in enumerate(line[1:]):
            if index+1>max_len:
                max_len=index+1

            if i not in w2i.keys():
                temp.append(w2i['unknown'])
            else:
                temp.append(w2i[i])
        X_train.append(temp)

X_train_padding=[]
for i in X_train:
    if len(i)<=max_len:
        X_train_padding.append(i+[-1]*(max_len-len(i)))


def matrix_max_len_re_label_dic():
    return embedded_matrix, max_len, re_label_dic

def data():
    return torch.tensor(X_train_padding).cuda(), torch.tensor(y_train).cuda(), max(y_train)


y_dev=[]
X_dev=[]
with open('/content/drive/My Drive/cnn_data/odp.wsdm.dev.all', 'r') as f:
    while True:
        line = f.readline().replace(',', '').split()
        if not line: break

        y_dev.append(int(line[0]))
        temp=[]
        for index, i in enumerate(line[1:]):
            if index + 1 > max_len:
                continue

            if i not in w2i.keys():
                temp.append(w2i['unknown'])
            else:
                temp.append(w2i[i])
        X_dev.append(temp)


X_dev_padding=[]
for i in X_dev:
    if len(i)<=max_len:
        X_dev_padding.append(i+[-1]*(max_len-len(i)))


print('train_data 수 : ', len(y_train))
print('dev_data 수 :', len(y_dev))


def data_dev():
    return torch.tensor(X_dev_padding).cuda(), torch.tensor(y_dev).cuda()