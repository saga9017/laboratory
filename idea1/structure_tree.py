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
view_depth=4


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

x=torch.randn(2, 5, 128)
y=torch.tensor([[1.0,2.0,3.0,4.0,5.0],
               [6.0, 7.0, 8.0, 9.0, 10.0]])
class test(nn.Module):
    def __init__(self):
        super().__init__()

        self.w0 = Parameter(torch.randn(view_depth+1, 5, 128))
        self.w1=Parameter(torch.randn(view_depth+1, 128,128))
        self.w6=Parameter(torch.randn(128,1))
        self.w7=Parameter(torch.randn(len(combination), len(combination)))

        self.layernorm1 = nn.ModuleList([LayerNorm(128)]*(view_depth+1)*len(combination))
        self.layernorm2 = nn.ModuleList([LayerNorm(128)]*(view_depth+1)*len(combination))

        self.layernorm3 = LayerNorm(len(combination))
        self.layernorm4 = LayerNorm(len(combination))

    def forward(self,x):
        result = []

        for index1, i in enumerate(combination):

            output = x.clone()
            for index2, (a,b) in enumerate(zip(i, self.w1)):
                if a == '0':
                    output=output+torch.mm(self.w0[index2], b)
                elif a == '1':
                    output=torch.matmul(output, b)
                    output=self.layernorm1[index1*(view_depth+1)+index2](output)
                elif a == '2':
                    output=torch.exp(output)
                    output=self.layernorm2[index1*(view_depth+1)+index2](output)
                elif a == '3':
                    output=F.relu(output)


            output=torch.matmul(output, self.w6)
            result.append(output)

        result=torch.cat(result, dim=2)
        result=self.layernorm3(result)
        mask7=1-(self.w7<=1e-7).float()
        new_w7=self.w7*mask7
        result=torch.matmul(result, new_w7)
        result = self.layernorm4(result)
        result=torch.sum(result, dim=2)
        return result, mask7

model=test()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(200):
    lrate = math.pow(1024, -0.5) * min(math.pow(epoch + 1, -0.5), (epoch + 1) * math.pow(10, -1.5))
    optimizer.param_groups[0]['lr'] = lrate
    optimizer.zero_grad()
    result, mask=model(x)
    loss=torch.mean((y-result)**2)
    loss.backward()
    model.w7.grad=model.w7.grad*mask
    optimizer.step()
    print('loss :', loss.item(), '       lrate :', optimizer.param_groups[0]['lr'])



