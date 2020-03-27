# skip-gram_hierarchical-softmax
import numpy as np
from datetime import datetime
import sys
import matplotlib.pyplot as plt


word=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
vocabulary_size=28
index_of_word={}

index_of_word['start']=0
index_of_word['end']=1



index=2
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

def softmax_grad(s):
    # Take the derivative of softmax element w.r.t the each logit which is usually Wi * X
    # input s is softmax value of the original input x.
    # s.shape = (1, n)
    # i.e. s = np.array([0.3, 0.7]), x = np.array([0, 1])
    # initialize the 2-D jacobian matrix.
    jacobian_m = np.diag(s)
    for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            if i == j:
                jacobian_m[i][j] = s[i] * (1-s[i])
            else:
                jacobian_m[i][j] = -s[i]*s[j]

    return jacobian_m


#train data 만들기
X_train=[]
y_train=[]

i=0
for _ in range(1000):
    X_train.append([index_of_word[j ]for j in word[i:i+10]])
    temp=word[i+1:i+11]
    temp.insert(0, 'start')
    y_train.append([index_of_word[j ]for j in temp])
    word.append(word.pop(0))

print(X_train)
print(y_train)


class RNN_attention:
    def __init__(self, word_dim, hidden_dim=100):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

        self.U_ = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
        self.W_ = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))

        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (word_dim, 2*hidden_dim))

        self.vU = np.zeros_like(self.U)
        self.vV = np.zeros_like(self.V)
        self.vW = np.zeros_like(self.W)
        self.mU = np.zeros_like(self.U)
        self.mV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W)

        self.vU_ = np.zeros_like(self.U_)
        self.vW_ = np.zeros_like(self.W_)
        self.mU_ = np.zeros_like(self.U_)
        self.mW_ = np.zeros_like(self.W_)

    def forward_propagation(self, x, y):
        # The total number of time steps
        T1 = len(x)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        h = np.zeros((T1 + 1, self.hidden_dim))
        h[-1] = np.zeros(self.hidden_dim)
        # The outputs at each time step. Again, we save them for later.
        # For each time step...
        for t in np.arange(T1):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            h[t] = np.tanh(self.U[:, x[t]] + self.W.dot(h[t - 1]))

        T2 = len(y)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = np.zeros((T2 + 1, self.hidden_dim))
        s[-1] = h[-2]
        # The outputs at each time step. Again, we save them for later.
        o = np.zeros((T2, vocabulary_size))
        result=np.zeros((T2 , self.hidden_dim*2))
        alpha=np.zeros((T2 , T1))
        # For each time step...
        for t in np.arange(T2):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            s[t] = np.tanh(self.U_[:, y[t]] + self.W_.dot(s[t - 1]))
            e=np.dot(h[:-1],s[t].T)
            alpha[t]=softmax(e)
            a=np.dot(alpha[t].T, h[:-1])
            result[t]=np.concatenate((a, s[t]), 0)
            o[t]=softmax(self.V.dot(result[t]))

        return [o, result, alpha, h, s]


    def bptt(self, x, y):
        T2 = len(y)
        y_=y[1:]
        y_.append(1)
        # Perform forward propagation
        o, result, alpha, h, s = self.forward_propagation(x, y)
        # We accumulate the gradients in these variables
        dLdU_ = np.zeros(self.U_.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW_ = np.zeros(self.W_.shape)
        dsnext = np.zeros_like(s[0])  # (hidden_size,1)
        dh_dot_minus_one=np.zeros_like(h[:-1])
        #print('np.arange(T)[::-1] :', np.arange(T)[::-1])
        # For each output backwards...
        for t in np.arange(T2)[::-1]:
            # Initial delta calculation

            dy = np.copy(o[t])  # shape (num_chars,1).  "dy" means "dloss/dy"
            dy[y_[t]] -= 1  # backprop into y. After taking the soft max in the input vector, subtract 1 from the value of the element corresponding to the correct label.
            dLdV += np.outer(dy, result[t])
            dresult=np.dot(dy, self.V)
            da=dresult[:int(len(dresult)/2)]
            ds_1=dresult[int(len(dresult)/2):]
            dalpha=np.dot(h[:-1], da)
            dh_dot_minus_one+=np.outer(alpha[t], da)
            de=dalpha*alpha[t]*(1-alpha[t])
            dh_dot_minus_one+=np.outer(de, s[t])
            ds_2=np.dot(de, h[:-1])
            dsraw = (1 - s[t] * s[t]) * (ds_1+ds_2+dsnext)
             # backprop through tanh nonlinearity #tanh'(x) = 1-tanh^2(x)
            dLdU_[:, y[t]] += dsraw
            dLdW_ += np.outer(dsraw, s[t - 1])
            dsnext = np.dot(self.W_.T, dsraw)

        dhnext=dsnext
        print(dhnext)

        T1 = len(x)

        dLdU = np.zeros(self.U.shape)
        dLdW = np.zeros(self.W.shape)

        for t in np.arange(T1)[::-1]:
            # Initial delta calculation
            dh=dh_dot_minus_one[t]+dhnext
            dhraw = (1 - h[t] * h[t]) * dh  # backprop through tanh nonlinearity #tanh'(x) = 1-tanh^2(x)
            dLdU[:, x[t]] += dhraw
            dLdW += np.outer(dhraw, h[t - 1])
            dhnext = np.dot(self.W.T, dhraw)


        return [dLdU, dLdW, dLdU_, dLdW_, dLdV]

    def predict(self, x):
        # Perform forward propagation and return index of the highest score
        T1 = len(x)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        h = np.zeros((T1 + 1, self.hidden_dim))
        h[-1] = np.zeros(self.hidden_dim)
        # The outputs at each time step. Again, we save them for later.
        # For each time step...
        for t in np.arange(T1):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            h[t] = np.tanh(self.U[:, x[t]] + self.W.dot(h[t - 1]))


        s_pre = h[-2]
        z=0
        pred=[]
        att=[]
        # For each time step...
        while z!=1:
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            s = np.tanh(self.U_[:, z] + self.W_.dot(s_pre))
            e = np.dot(h[:-1], s.T)
            alpha = softmax(e)
            print(np.argmax(alpha))
            a = np.dot(alpha.T, h[:-1])
            result = np.concatenate((a, s), 0)
            o = softmax(self.V.dot(result))
            z=np.argmax(o)
            pred.append(z)
            att.append(alpha)
            s_pre=s

        return pred, att

    def calculate_total_loss(self, x, y):
        L = 0
        # For each sentence...
        for i in np.arange(len(y)):
            o, result, alpha, h, s  = self.forward_propagation(x[i], y[i])
            y_=y[i][1:]
            y_.append(1)
            # We only care about our prediction of the "correct" words
            correct_word_predictions = o[np.arange(len(y[i])), y_]
            #print('correct_word_predictions :', correct_word_predictions)
            # Add to the loss based on how off we were
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L


    def calculate_loss(self, x, y):
        # Divide the total loss by the number of training examples
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x, y) / N




    # Performs one step of SGD.
    def numpy_sdg_step(self, x, y, learning_rate):
        # Calculate the gradients
        dLdU, dLdW, dLdU_, dLdW_, dLdV = self.bptt(x, y)
        # Change parameters according to gradients and learning rate

        self.mU = self.beta1 * self.mU + (1 - self.beta1) * dLdU
        self.mV = self.beta1 * self.mV + (1 - self.beta1) * dLdV
        self.mW = self.beta1 * self.mW + (1 - self.beta1) * dLdW
        self.mU_ = self.beta1 * self.mU_ + (1 - self.beta1) * dLdU_
        self.mW_ = self.beta1 * self.mW_ + (1 - self.beta1) * dLdW_

        self.vU = self.beta2 * self.vU + (1 - self.beta2) * (dLdU ** 2)
        self.vV = self.beta2 * self.vV + (1 - self.beta2) * (dLdV ** 2)
        self.vW = self.beta2 * self.vW + (1 - self.beta2) * (dLdW ** 2)
        self.vU_ = self.beta2 * self.vU_ + (1 - self.beta2) * (dLdU_ ** 2)
        self.vW_ = self.beta2 * self.vW_ + (1 - self.beta2) * (dLdW_ ** 2)

        self.U += -learning_rate * self.mU / (np.sqrt(self.vU) + self.eps)
        self.V += -learning_rate * self.mV / (np.sqrt(self.vV) + self.eps)
        self.W += -learning_rate * self.mW / (np.sqrt(self.vW) + self.eps)
        self.U_ += -learning_rate * self.mU_ / (np.sqrt(self.vU_) + self.eps)
        self.W_ += -learning_rate * self.mW_ / (np.sqrt(self.vW_) + self.eps)




    def train_with_sgd(self, X_train, y_train, learning_rate=0.0025, nepoch=100, evaluate_loss_after=5):
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
                    pass
                    #learning_rate = learning_rate * 0.5
                    #print("Setting learning rate to %f" % learning_rate)
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
model = RNN_attention(vocabulary_size)
losses = model.train_with_sgd(X_train[:100], y_train[:100], nepoch=250, evaluate_loss_after=25)


x=['c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
y=[]
for word in x:
    if word not in index_of_word.keys():
        y.append(0)
    else:
        y.append(index_of_word[word])

print('y :', y)
print('predict :', model.predict(y)[0])
print('predict :', [word_of_index[i] for i in model.predict(y)[0]])