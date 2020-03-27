import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch.nn.parameter import Parameter
import torch.nn.functional  as F
from torch.optim.lr_scheduler import LambdaLR
import math

a=torch.randn(2, 5, 4)
w1=Parameter(torch.randn(4,5))
y=torch.tensor([5, 6])


operation_dic={0:'+', 1:'*', 2:'exp', 3:'relu'}
operation_list=[]

#print(torch.matmul(a, w1))
view_depth=5


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


class Tree():
    def __init__(self, index):
        self.index=str(index)
        self.child=[]

    def aug(self):
        self.child=[Tree(self.index+'0'), Tree(self.index+'1'), Tree(self.index+'2'), Tree(self.index+'3')]

tree=Tree(0)
tree.aug()
depth=0
temp_trees=tree.child
while depth!=view_depth:
    depth+=1
    temp=[]
    for i in temp_trees:
        i.aug()
        temp.extend(i.child)
    temp_trees=temp

combination=[]
print('tree node 개수 :',len(temp_trees))
for i in temp_trees:
    if '00' in i.index[1:]:
        pass
    else:
        if '33' in i.index[1:]:
            pass
        else:
            combination.append(i.index[1:])


print('combination 수 :', len(combination))
print(combination)

x=torch.randn(5, 128)
y=torch.tensor([1.0,2.0,3.0,4.0,5.0])
class test(nn.Module):
    def __init__(self):
        super().__init__()

        self.w0 = nn.ModuleList([nn.Linear(128, 128)])
        self.w1=nn.ModuleList([nn.Linear(128, 5)])
        self.w2=nn.Linear(5,1)
        self.layernorm3 = LayerNorm(128)

    def forward(self,x):
        result=x.clone()
        result=self.w0[0](result)
        result=self.layernorm3(result)
        #result=F.relu(result)
        result=self.w1[0](result)
        result=self.w2(result)
        return result

model=test()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(200):
    optimizer.zero_grad()
    result=model(x)
    print(result)
    loss=torch.mean((y-result)**2)
    loss.backward()
    optimizer.step()
    print(loss)



