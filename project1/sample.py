import pickle
import numpy as np

with open('w2v_lower.pkl', 'rb') as fin:
    embedded = pickle.load(fin)


embedded_matrix=np.concatenate((embedded[0], np.random.randn(1,300)), 0)
w2i=embedded[1]

##############단어하나당 embedding 300####################
#print(w2i.keys())

y_train=[]
X_train=[]

max_len=1
with open('cnn_data/odp.wsdm.train.all', 'r') as f:
    while True:
        line = f.readline().replace(',', '').split()
        if not line: break
        y_train.append(int(line[0]))
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
    if len(i)<max_len:
        X_train_padding.append(i+[-1]*(max_len-len(i)))
