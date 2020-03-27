# skip-gram_hierarchical-softmax
import numpy as np
from datetime import datetime
import sys


word=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
vocabulary_size=26
index_of_word={}


index=0
for alphabet in word:
    index_of_word[alphabet]=index
    index+=1

print(index_of_word)
word_of_index={y:x for x,y in index_of_word.items()}


def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


#train data 만들기
X_train=[]
y_train=[]

i=0
for _ in range(1000):
    X_train.append([index_of_word[j ]for j in word[i:i+10]])
    y_train.append([index_of_word[j ]for j in word[i+1:i+11]])
    word.append(word.pop(0))

print(X_train)
print(y_train)


class RNNNumpy:
    def __init__(self, word_dim, hidden_dim=100):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

    def forward_propagation(self, x):
        # The total number of time steps
        T = len(x)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        # The outputs at each time step. Again, we save them for later.
        o = np.zeros((T, self.word_dim))
        # For each time step...
        for t in np.arange(T):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t - 1]))
            o[t] = softmax(self.V.dot(s[t]))
        return [o, s]


    def predict(self, x):
        # Perform forward propagation and return index of the highest score
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)

    def calculate_total_loss(self, x, y):
        L = 0
        # For each sentence...
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            # We only care about our prediction of the "correct" words
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            #print('correct_word_predictions :', correct_word_predictions)
            # Add to the loss based on how off we were
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L


    def calculate_loss(self, x, y):
        # Divide the total loss by the number of training examples
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x, y) / N

    def bptt(self, x, y):
        T = len(y)
        # Perform forward propagation
        o, s = self.forward_propagation(x)
        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        dhnext = np.zeros_like(s[0])  # (hidden_size,1)

        #print('np.arange(T)[::-1] :', np.arange(T)[::-1])
        # For each output backwards...
        for t in np.arange(T)[::-1]:
            # Initial delta calculation

            dy = np.copy(o[t])  # shape (num_chars,1).  "dy" means "dloss/dy"
            dy[y[t]] -= 1  # backprop into y. After taking the soft max in the input vector, subtract 1 from the value of the element corresponding to the correct label.
            dLdV += np.outer(dy, s[t])
            dh = np.dot(self.V.T, dy) + dhnext  # backprop into h.
            dhraw = (1 - s[t] * s[t]) * dh  # backprop through tanh nonlinearity #tanh'(x) = 1-tanh^2(x)
            dLdU += np.outer(dhraw, x[t])
            dLdW += np.outer(dhraw, s[t - 1])
            dhnext = np.dot(self.W.T, dhraw)

        for dparam in [dLdV , dLdU, dLdW]:
            np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients.

        return [dLdU, dLdV, dLdW]


    # Performs one step of SGD.
    def numpy_sdg_step(self, x, y, learning_rate):
        # Calculate the gradients
        dLdU, dLdV, dLdW = self.bptt(x, y)
        # Change parameters according to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW


    def train_with_sgd(self, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
        # We keep track of the losses so we can plot them later
        losses = []
        num_examples_seen = 0
        for epoch in range(nepoch):
            # Optionally evaluate the loss
            if (epoch % evaluate_loss_after == 0):
                loss = self.calculate_loss(X_train, y_train)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
                # Adjust the learning rate if loss increases
                if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                    learning_rate = learning_rate * 0.5
                    print("Setting learning rate to %f" % learning_rate)
                sys.stdout.flush()
            # For each training example...
            for i in range(len(y_train)):
                # One SGD step
                self.numpy_sdg_step(X_train[i], y_train[i], learning_rate)
                #print(X_train[i])
                #print(self.predict(X_train[i]))
                num_examples_seen += 1


np.random.seed(10)
# Train on a small subset of the data to see what happens
model = RNNNumpy(vocabulary_size)
losses = model.train_with_sgd(X_train[:100], y_train[:100], nepoch=10, evaluate_loss_after=1)


x=['c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
y=[]
for word in x:
    if word not in index_of_word.keys():
        y.append(0)
    else:
        y.append(index_of_word[word])

print('y :', y)
print('predict :', model.predict(y))
print('predict :', word_of_index[model.predict(y)[-1]])