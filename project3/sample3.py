import torch
import torch.nn as nn
import numpy as np
import pickle
print(torch.sigmoid(torch.tensor(1.0)))

def sigmoid(x):
    return 1/(1+torch.exp(-x))

print(sigmoid(torch.tensor(1.0)))


print((torch.tensor([1,0])!=2).float())


class Test(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1=nn.Linear(3,5)
        self.linear2=nn.Linear(5,6)

    def forward(self,x):
        return


test=Test()
optimizer = torch.optim.Adam(list(test.linear1.parameters())+list(test.linear2.parameters()), betas=(0.9, 0.999), eps=1e-9, lr=1e-4)

a=torch.tensor([[1,2,3,4,5], [6,1,2,7,8]])
print(a.shape)
print(torch.max(a, dim=0)[0])

list=[]
a=np.array([[1,2,3,4,5], [1,2,3,4,5]])
b=np.array([[2,3,4,5,6], [7,8,9,10,11]])

list.append(a)
list.append(b)

c=np.concatenate(list, axis=0)
print(a.shape)
print(b.shape)
print(c)
print(c.shape)

print('=====================================================')
a=np.array([[[1,2,3,4,5], [1,2,3,4,5]], [[2,3,4,5,6], [7,8,9,10,11]]])
b=np.array([[[2,3,4,5,6], [7,8,9,10,11]], [[2,3,4,5,6], [7,8,9,10,11]], [[2,3,4,5,6], [7,8,9,10,11]]])

print(a.shape)
print(b.shape)

list=[]

list.extend(a.tolist())
list.extend(b.tolist())
print(np.array(list).shape)


##with open('bert_text_test', 'rb') as f:
##    data = pickle.load(f) # 단 한줄씩 읽어옴

#print(np.array(data).shape)


a=[1,2,3,4,5]
print(a)

del a[:]


a1=np.load("project3_transformer_data/bert_text_train4.npy")
for index, i in enumerate(a1):
    np.save("project3_transformer_text_data/bert_text_train"+str(index+60000), i)
"""""
#a2=np.load("project3_transformer_data/bert_text_train2.npy")
#a3=np.load("project3_transformer_data/bert_text_train3.npy")
#a4=np.load("project3_transformer_data/bert_text_train4.npy")

print(a1.shape)
"""""

a=np.array([1,2,3,4,5])
print(a.shape)
print('==============')
print(np.expand_dims(a, axis=0).shape)
