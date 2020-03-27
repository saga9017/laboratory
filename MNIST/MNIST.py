import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from matplotlib import pyplot as plt


train_data = dsets.MNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True)
test_data  = dsets.MNIST(root='data/', train=False, transform=transforms.ToTensor(), download=True)

print('number of training data: ', len(train_data))
print('number of test data: ', len(test_data))

image, label = train_data[0]


plt.imshow(image.squeeze().numpy(), cmap='gray')
plt.title('%i' % label)
plt.show()