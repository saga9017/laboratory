import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

from openpyxl import load_workbook



def data_load(filename):
    load_wb = load_workbook(filename, data_only=True)

    X=[]
    y=[]
    load_ws = load_wb['Sheet']
    for row in load_ws.rows:
        X.append([row[0].value, row[1].value])
        y.append(row[2].value)

    X=torch.tensor(X[1:])
    y=torch.tensor(y[1:]).float()

    return X, y


X_train, y_train=data_load('train_set.xlsx')
X_val, y_val=data_load('val_set.xlsx')
X_test, y_test=data_load('test_set.xlsx')

def scatter_plot(X, y):
    plt.scatter(X[y==0, 0], X[y==0, 1], color='red')
    plt.scatter(X[y==1, 0], X[y==1, 1], color='blue')
    plt.show()

scatter_plot(X_train, y_train)
scatter_plot(X_val, y_val)
scatter_plot(X_test, y_test)



class Model(nn.Module):

 def __init__(self, input_size, H1, output_size):
     super().__init__()
     self.linear1 = nn.Linear(input_size, H1)
     self.linear2= nn.Linear(H1, output_size)

 def forward(self, x):
     x = torch.sigmoid(self.linear1(x))
     x = torch.sigmoid(self.linear2(x))
     return x

 def predict(self, x):
     return self.forward(x) >= 0.5


model = Model(2, 4, 1)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def cal_score(X, y):
    y_pred=model.predict(X)
    score=float(torch.sum(y_pred.squeeze(-1)==y.byte()))/y.shape[0]

    return score


epochs = 1000
losses = []

#train step
for i in range(epochs):
    y_pred = model.forward(X_train)
    loss = criterion(y_pred, y_train)

    #print("epochs:", i, "loss:", loss.item())

    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #calculate validation score
    if i% 10==0:
        print(cal_score(X_val, y_val))


print('test score :', cal_score(X_test, y_test))
plt.plot(range(epochs), losses)
plt.show()

def plot_decision_boundray(X):
    x_span = np.linspace(min(X[:, 0]), max(X[:, 0]))
    y_span = np.linspace(min(X[:, 1]), max(X[:, 1]))

    xx, yy = np.meshgrid(x_span, y_span)

    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()]).float()

    pred_func = model.forward(grid)

    z = pred_func.view(xx.shape).detach().numpy()

    plt.contourf(xx, yy, z)
    plt.show()


plot_decision_boundray(X_train)
