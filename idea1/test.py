import torch
import torch.nn as nn
import torch.nn.functional  as F


softmax=nn.Softmax(-1)
batchnorm=nn.BatchNorm1d(5)
x=torch.randn(5,5)

y=softmax(x)
y_=softmax(batchnorm(x))

print(x)
print(y)
print(y_)
