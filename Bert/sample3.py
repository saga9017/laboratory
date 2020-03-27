score=0
for _ in range(52):
    score+=1


print(score/10)


import torch
values, indices=torch.topk(torch.tensor([[0.01, 0.5, 0.3, 0.4], [0.01, 0.5, 0.3, 0.4]]),2)

print(values)
print(indices)


list=[0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1]
print(list.index(1))