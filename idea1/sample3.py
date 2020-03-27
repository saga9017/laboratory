import torch
import sys
x= torch.tensor([[1,2,3,4,5],
                 [6,7,8,9,10]])

y= torch.tensor([[1,1,1,1,1],
                 [1,1,1,1,1]])


z=[x, y]

for i in z:
    pass

standard={}
for index, param in enumerate(y):
    standard[index]=param


for index, param in enumerate(x):
    param.data+=-param.data+standard[index]

print(x)

for index, param in enumerate(y):
    param.data-=100

print(x)


print(torch.randn(3))
