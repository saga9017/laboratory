from openpyxl import load_workbook
import torch
import torch.nn.functional  as F
import torch.nn as nn


load_wb = load_workbook("project2.xlsx", data_only=True)
load_ws = load_wb['Sheet1']

used_index=[3, 7, 8, 9, 10 ,11, 12, 13, 14]

#data 전처리
rows=[]
X_train=[]
y_train=[]
for index2, row in enumerate(load_ws.rows):
    if index2==0:
        continue
    temp=[]
    for index, data in enumerate(row):
        if index in used_index:
            if index==3:
                if data.value=='Y':
                    temp.append(1)
                else:
                    temp.append(0)
            elif index==7:
                temp.append(int(data.value/10))
            elif index==8:
                if data.value=='남':
                    temp.append(1)
                else:
                    temp.append(0)
            elif index==14:
                if data.value=='R':
                    temp.append(2)
                elif data.value=='A':
                    temp.append(1)
                else:
                    temp.append(0)
            elif index==10:
                if data.value<20000200:
                    temp.append(1)
                else:
                    temp.append(0)
            elif index==11:
                if data.value=='-':
                    temp.append(0)
                else:
                    if data.value<5000000:
                        temp.append(2)
                    else:
                        temp.append(1)
            elif index==12:
                if data.value == '-':
                    temp.append(0)
                else:
                    if data.value < 100000000:
                        temp.append(2)
                    else:
                        temp.append(1)
            elif index==13:
                if data.value=='-':
                    temp.append(0)
                else:
                    if data.value<200000000:
                        temp.append(2)
                    else:
                        temp.append(1)
            else:
                if data.value=='-':
                    temp.append(0)
                else:
                    temp.append(data.value)
    X_train.append(temp[:-1])
    y_train.append(temp[-1])


print('X_train :', X_train)
print('y_train :', y_train)


class classification(nn.Module):
    def __init__(self):  # default=512
        # Assign instance variables
        super().__init__()
        self.embeddings=nn.ModuleList([nn.Embedding(2, 128), nn.Embedding(9, 128), nn.Embedding(2, 128), nn.Embedding(21421, 128),
                                       nn.Embedding(2, 128), nn.Embedding(3, 128), nn.Embedding(3, 128), nn.Embedding(3, 128)])
        for i in self.embeddings:
            i.weight.data.uniform_(-0.1, 0.1)

        self.linear1 = nn.Linear(128, 1)
        self.linear2 = nn.Linear(8, 3)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)

        self.softmax=nn.Softmax(-1)

    def forward(self, x):
        stack=[]
        for a,b  in zip(x, self.embeddings):
            stack.append(b(a).unsqueeze(0))
        result=torch.cat(stack, dim=0)


        result=F.relu(self.linear1(result)).squeeze(-1)
        result=self.softmax(self.linear2(result))

        return result

    def cal_loss(self, x, y):
        result=self.forward(x)
        loss=-torch.log(result)[y]
        return loss

    def sdg_step(self, x, y, optimizer):
        optimizer.zero_grad()
        loss =self.cal_loss(torch.tensor(x), torch.tensor(y))
        loss.backward()
        optimizer.step()

        return loss

    def train_step(self, X, Y, nepoch=10):
        optimizer = torch.optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=0.1)
        for epoch in range(nepoch):
            Loss=0
            # For each training example...
            for i in range(len(X)):
                # One SGD step
                Loss += self.sdg_step(X[i], Y[i], optimizer).item()

            print('Train loss : ', Loss/len(X))



model=classification()
model.train_step(X_train, y_train)
