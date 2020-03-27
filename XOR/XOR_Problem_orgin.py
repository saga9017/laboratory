import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


# XOR 문제를 해결하기 위해 dataset 만들기.
X_data=torch.tensor([[0,0], [0,1], [1,0], [1,1]]).float()
y_data=torch.tensor([0,1,1,0]).float()

"""
NPUT	OUTPUT
A	B	A XOR B
0	0	0
0	1	1
1	0	1
1	1	0
"""


class Model(nn.Module):

 def __init__(self, input_size, H1, output_size):
     super().__init__()
     self.linear1 = nn.Linear(input_size, H1)
     self.linear2 = nn.Linear(H1, output_size)

 def forward(self, x):

     x = torch.sigmoid(self.linear1(x))
     x = torch.sigmoid(self.linear2(x))
     return x

 def predict(self, x):
     return self.forward(x) >= 0.5

model = Model(2, 2, 1)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 2000
losses = []

#batch train step
for i in range(epochs):
    y_pred = model.forward(X_data)
    loss = criterion(y_pred, y_data)

    print("epochs:", i, "loss:", loss.item())

    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def cal_score(X, y):
    y_pred=model.predict(X)
    score=float(torch.sum(y_pred.squeeze(-1)==y.byte()))/y.shape[0]

    return score

print('test score :', cal_score(X_data, y_data))
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


plot_decision_boundray(X_data)