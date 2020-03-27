from __future__ import print_function

import torch

class TwoLayerNet(object):
  """
  two-layer-perceptron.
  Input dimension : N
  Hidden layer dimension : H
  Output dimension : C

  Softmax loss function을 활용해 네트워크를 학습시킬 것입니다.
  Hidden layer의 activation function으로는 ReLU를 사용합니다.

  정리하자면, 네트워크는 다음과 같은 구조를 갖습니다.

  input - linear layer - ReLU - linear layer - output
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    모델을 초기화하며 weight는 작은 랜덤값, bias는 0으로 초기화됩니다.
    Weight와 bias는 self.params라는 dictionary에 저장됩니다.

    W1: 첫 번째 layer의 weight; (D, H)
    b1: 첫 번째 layer의 biase; (H,)
    W2: 두 번째 layer의 weight; (H, C)
    b2: 두 번째 layer의 biase; (C,)

    Inputs:
    - input_size: input data의 dimension.
    - hidden_size: hidden layer의 neuron(node) 개수.
    - output_size: output dimesion.
    """
    self.params = {}
    self.params['W1'] = std * torch.randn(input_size, hidden_size)
    self.params['b1'] = torch.zeros(hidden_size)
    self.params['W2'] = std * torch.randn(hidden_size, output_size)
    self.params['b2'] = torch.zeros(output_size)

  def loss(self, X, y=None):
    """
    Neural network의 loss와 gradient를 계산합니다.

    Inputs:
    - X: Input data. shape (N, D). 각각의 X[i]가 하나의 training sample이며 총 N개의 sample이 input으로 주어짐.
    - y: Training label 벡터. y[i]는 X[i]에 대한 정수값의 label.
      y가 주어질 경우 loss와 gradient를 반환하며 y가 주어지지 않으면 output을 반환

    Returns:
    y가 주어지지 않으면, shape (N, C)인 score matrix 반환
    scores[i, c]는 input X[i]에 대한 class c의 score

    y가 주어지면 (loss, grads) tuple 반환
    loss: training batch에 대한 loss (scalar)
    grads: {parameter 이름: gradient} 형태의 dictionary (self.params와 같은 키여야 함)
    """
    # Dictionary에서 weight와 bias 불러오기
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.size()

    # Forward path 계산
    scores = None
    #############################################################################
    # TODO: Forward path를 수행하고, 'scores'에 결과값을 저장 (shape : (N, C))  #
    #         input - linear layer - ReLU - linear layer - output             #
    #############################################################################
    hidden = X.mm(W1)+b1
    hidden_relu = hidden.clamp(min=0)
    scores = hidden_relu.mm(W2)+b2
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # 정답(target)이 주어지지 않은 경우 점수를 리턴하고 종료
    if y is None:
      return scores

    # Loss 계산
    loss = None
    e = torch.exp(scores)
    softmax = e / torch.sum(e, dim=1, keepdim=True)
    #############################################################################
    #       TODO: Output을 이용하여 loss값 계산하고, 'loss'에 저장(scalar)        #
    #                loss function : negative log likelihood                    #
    #              'softmax' 변수에 저장된 softmax값을 이용해서 계산              #
    #         'y'는 정답 index를 가리키며 정답 확률에 -log 적용하여 평균           #
    #############################################################################
    prob=[]
    j=0
    for i in y:
      prob.append(softmax[j][i])
      j+=1

    loss=torch.mean(-torch.log(torch.FloatTensor(prob)))

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward path(Gradient 계산) 구현
    grads = {}
    #############################################################################
    # TODO: Weight와 bias에 대한 gradient를 계산하고 'grads' dictionary에 저장   #
    #               dictionary의 key는 self.params와 동일하게 설정.             #
    #          grads['W1']는 self.params['W1']과 같은 shape를 가져야 함.        #
    #              softmax의 gradient부터 차근차근 구해나가도록 함.              #
    #############################################################################
    z=softmax
    j = 0
    for i in y:
      z[j][i]-=1
      j+=1

    grad_y_pred = z
    grads['W2'] = hidden_relu.t().mm(grad_y_pred)
    grads['b2'] = torch.ones(1,N).mm(grad_y_pred)
    grad_hidden_relu = grad_y_pred.mm(W2.t())
    grad_hidden = grad_hidden_relu.clone()
    grad_hidden[hidden < 0] = 0
    grads['W1'] = X.t().mm(grad_hidden)
    grads['b1'] = torch.ones(1,N).mm(grad_hidden)

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return loss, grads

  def train(self, X, y,
            learning_rate=1e-3, learning_rate_decay=0.95,
            num_iters=100,
            batch_size=200, verbose=False):
    """
    SGD를 이용한 neural network training

    Inputs:
    - X: shape (N, D)의 numpy array (training data)
    - y: shape (N,)의 numpy array(training labels; y[i] = c
                                  c는 X[i]의 label, 0 <= c < C)
    - learning_rate: Scalar learning rate
    - num_iters: Number of steps
    - batch_size: Number of training examples in a mini-batch.
    - verbose: true일 경우 progress 출력
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)
    # SGD를 이용한 optimization
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
      loss, grads = self.loss(X, y=y)
      loss_history.append(loss)

      #########################################################################
      # TODO: 'grads' dictionary에서 gradient를 불러와 SGD update 수행        #
      #########################################################################
      self.params['W1'] = self.params['W1'] - learning_rate * grads['W1']
      self.params['b1'] = self.params['b1'] - learning_rate * grads['b1']
      self.params['W2'] = self.params['W2'] - learning_rate * grads['W2']
      self.params['b2'] = self.params['b2'] - learning_rate * grads['b2']
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))


      if it % iterations_per_epoch == 0:
        # Accuracy
        train_acc = (self.predict(X) == y).float().mean()
        train_acc_history.append(train_acc)

        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    values, index=torch.max(self.loss(X), 1)
    return index


