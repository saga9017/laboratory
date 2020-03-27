from transformer_model import transformer
from preprocessing import data
import torch

X_train, y_train, label_number, label_dic= data()

model = transformer(label_number)
model.load_state_dict(torch.load('saved_model'))

print(model.predict(X_train[:3]))


with open('cnn_data/odp.wsdm.train.all', 'r') as f:
    while True:
        line = f.readline().replace(',', '').split()
        if not line: break
        if int(line[0]) not in label_dic:
            y_train.append(-1)
        else:
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