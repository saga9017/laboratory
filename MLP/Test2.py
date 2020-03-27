# A bit of setup

import torch

from NN import TwoLayerNet

# Create a small net and some toy data to check your implementations.
# Note that we set the random seed for repeatable experiments.

input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5

def init_toy_model():
    torch.manual_seed(0)
    return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)

def init_toy_data():
    torch.manual_seed(1)
    X = 10 * torch.randn(num_inputs, input_size)
    y = torch.LongTensor([0, 1, 2, 2, 1])
    return X, y

net = init_toy_model()
X, y = init_toy_data()

scores = net.loss(X)
print('Your scores:')
print(scores)
print()
print('correct scores:')
correct_scores = torch.Tensor(
  [[ 0.24617445,  0.1261572,   1.1627575 ],
 [ 0.18364899, -0.0675799,  -0.21310908],
 [-0.2075074,  -0.12525336, -0.06508598],
 [ 0.08643292,  0.07172455,  0.2353122 ],
 [ 0.8219606,  -0.32560882, -0.77807254]]
)
print(correct_scores)
print()

print('Difference between your scores and correct scores:')
print(torch.sum(torch.abs(scores - correct_scores)))

loss, _ = net.loss(X, y)
correct_loss = 1.2444149

print('Difference between your loss and correct loss:')
print(torch.sum(torch.abs(loss - correct_loss)))

loss, grads = net.loss(X, y)

results = net.train(X, y, 0.05)
print("Train acc: %f -> %f\nTrain loss: %f -> %f" % (results['train_acc_history'][0], results['train_acc_history'][-1]
                                                , results['loss_history'][0],results['loss_history'][-1]))