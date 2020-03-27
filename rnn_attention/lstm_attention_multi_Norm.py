# skip-gram_hierarchical-softmax
import numpy as np
from datetime import datetime
import sys
import unicodedata
import re
import random
from random import uniform


view_sentence_len=1000

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  #  SOS 와 EOS 단어 숫자 포함

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1



def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# 소문자, 다듬기, 그리고 문자가 아닌 문자 제거


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines

    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')



    # 모든 줄을 쌍으로 분리하고 정규화 하십시오
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    pairs=pairs[:view_sentence_len]


    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH \
         #  and p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def batchnorm_forward(X, gamma, beta):
    mu = np.mean(X, axis=0)
    var = np.var(X, axis=0)

    X_norm = (X - mu) / np.sqrt(var + 1e-8)
    out = gamma * X_norm + beta

    cache = (X, X_norm, mu, var, gamma, beta)

    return out, cache, mu, var


def batchnorm_backward(dout, cache):
    X, X_norm, mu, var, gamma, beta = cache

    N= X.shape

    X_mu = X - mu
    std_inv = 1. / np.sqrt(var + 1e-8)

    dX_norm = dout * gamma
    dvar = np.sum(dX_norm * X_mu, axis=0) * -.5 * std_inv**3
    dmu = np.sum(dX_norm * -std_inv, axis=0) + dvar * np.mean(-2. * X_mu, axis=0)

    dX = (dX_norm * std_inv) + (dvar * 2 * X_mu / N) + (dmu / N)
    dgamma = np.sum(dout * X_norm, axis=0)
    dbeta = np.sum(dout, axis=0)

    return dX, dgamma, dbeta


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print('pair:', random.choice(pairs))

pairs=np.array(pairs)
print(pairs[:,0][:100])
print(pairs[:,1][:100])

X_train=[]
y_train=[]

for sentence in pairs[:,0]:
    temp=[]
    for word in sentence.split():
        temp.append(input_lang.word2index[word])
    X_train.append(temp)

for sentence in pairs[:,1]:
    temp=[]
    for word in sentence.split():
        temp.append(output_lang.word2index[word])
    temp.insert(0, 0)
    y_train.append(temp)

print(X_train)
print(y_train)

class LSTM_attention_multi:
    def __init__(self, word_dim1, word_dim2, hidden_dim=128):
        # Assign instance variables
        self.word_dim1 = word_dim1
        self.word_dim2 = word_dim2
        self.hidden_dim = hidden_dim
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.gamma = 1
        self.beta = 0
        self.mgamma = 0
        self.mbeta = 0
        self.vgamma = 0
        self.vbeta = 0
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./word_dim1), np.sqrt(1./word_dim1), (4*hidden_dim, word_dim1))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (4*hidden_dim, hidden_dim))

        self.U_ = np.random.uniform(-np.sqrt(1. / word_dim2), np.sqrt(1. / word_dim2), (4*hidden_dim, word_dim2))
        self.W_ = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (4*hidden_dim, hidden_dim))

        self.new_W=np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (word_dim2, 2*hidden_dim))

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

        self.vnew_W=np.zeros_like(self.new_W)
        self.mnew_W=np.zeros_like(self.new_W)

    def forward_propagation(self, x, y):
        # The total number of time steps
        T1 = len(x)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        h = np.zeros((T1 + 1, self.hidden_dim))
        h[-1] = np.zeros(self.hidden_dim)
        c = np.zeros((T1 + 1, self.hidden_dim))
        o = np.zeros((T1, self.hidden_dim))
        i = np.zeros((T1, self.hidden_dim))
        f = np.zeros((T1, self.hidden_dim))
        g = np.zeros((T1, self.hidden_dim))
        # The outputs at each time step. Again, we save them for later.
        # For each time step...
        H = self.hidden_dim
        for t in np.arange(T1):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            temp = self.U[:, x[t]] + self.W.dot(h[t - 1])
            i[t] = sigmoid(temp[:H])
            f[t] = sigmoid(temp[H:2 * H])
            o[t] = sigmoid(temp[2 * H:3 * H])
            g[t] = np.tanh(temp[3 * H:])
            c[t] = f[t] * (c[t - 1]) + i[t] * (g[t])
            h[t] = o[t] * np.tanh(c[t])


        T2 = len(y)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = np.zeros((T2 + 1, self.hidden_dim))
        s[-1] = h[-2]
        c_ = np.zeros((T2 + 1, self.hidden_dim))
        c_[-1] = c[-2]
        o_ = np.zeros((T2, self.hidden_dim))
        i_ = np.zeros((T2, self.hidden_dim))
        f_ = np.zeros((T2, self.hidden_dim))
        g_ = np.zeros((T2, self.hidden_dim))
        # The outputs at each time step. Again, we save them for later.
        output = np.zeros((T2, self.word_dim2))
        result=np.zeros((T2 , self.hidden_dim*2))
        alpha=np.zeros((T2 , T1))
        e_pre=np.zeros((T2, T1, self.hidden_dim))
        e_cache={}
        # For each time step...
        for t in np.arange(T2):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            temp_ = self.U_[:, y[t]] + self.W_.dot(s[t - 1])
            i_[t] = sigmoid(temp_[:H])
            f_[t] = sigmoid(temp_[H:2 * H])
            o_[t] = sigmoid(temp_[2 * H:3 * H])
            g_[t] = np.tanh(temp_[3 * H:])
            c_[t] = f_[t] * (c_[t - 1]) + i_[t] * (g_[t])
            s[t] = o_[t] * np.tanh(c_[t])
            e_pre[t]=np.dot(h[:-1],self.new_W)
            e=np.dot(e_pre[t],s[t].T)
            # batch_normalization
            e_, e_cache[t], mu, var = batchnorm_forward(e, self.gamma, self.beta)
            alpha[t]=softmax(e_)
            a=np.dot(alpha[t].T, h[:-1])
            result[t]=np.concatenate((a, s[t]), 0)
            output[t]=softmax(self.V.dot(result[t]))

        return [output, result, alpha, h, s, c, c_ , o, i, f, g, o_, i_, f_, g_, e_pre, e_cache]


    def bptt(self, x, y):
        T2 = len(y)
        y_=y[1:]
        y_.append(1)
        # Perform forward propagation
        output, result, alpha, h, s, c, c_ , o, i, f, g, o_, i_, f_, g_, e_pre, e_cache = self.forward_propagation(x, y)
        # We accumulate the gradients in these variables
        dLdU_ = np.zeros(self.U_.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW_ = np.zeros(self.W_.shape)
        dLdnew_W=np.zeros(self.new_W.shape)
        dsnext = np.zeros_like(s[0])  # (hidden_size,1)
        dh_dot_minus_one=np.zeros_like(h[:-1])
        dcnext_ = np.zeros_like(c_[0])
        #print('np.arange(T)[::-1] :', np.arange(T)[::-1])
        # For each output backwards...
        for t in np.arange(T2)[::-1]:
            # Initial delta calculation

            dy = np.copy(output[t])  # shape (num_chars,1).  "dy" means "dloss/dy"
            dy[y_[t]] -= 1  # backprop into y. After taking the soft max in the input vector, subtract 1 from the value of the element corresponding to the correct label.
            dLdV += np.outer(dy, result[t])
            dresult=np.dot(dy, self.V)
            da=dresult[:int(len(dresult)/2)]
            ds_1=dresult[int(len(dresult)/2):]
            dalpha=np.dot(h[:-1], da)
            dh_dot_minus_one += np.outer(alpha[t], da)
            de_=dalpha*alpha[t]*(1-alpha[t])
            de, dgamma, dbeta = batchnorm_backward(de_, e_cache[t])
            ####################################################################
            de_pre=np.outer(de, s[t])
            ds_2 = np.dot(de, e_pre[t])
            dLdnew_W+=np.dot(de_pre.T, h[:-1])
            dh_dot_minus_one += np.dot(de_pre, self.new_W)
            ds = (1 - s[t] * s[t]) * (ds_1+ds_2+dsnext)
            dc_ = dcnext_ + (1 - np.tanh(c_[t]) * np.tanh(c_[t])) * ds * o_[t]  # backprop through tanh nonlinearity
            dcnext_ = dc_ * f_[t]
            di_ = dc_ * g_[t]
            df_ = dc_ * c_[t - 1]
            do_ = ds * np.tanh(c_[t])
            dg_ = dc_ * i_[t]
            ddi_ = (1 - i_[t]) * i_[t] * di_
            ddf_ = (1 - f_[t]) * f_[t] * df_
            ddo_ = (1 - o_[t]) * o_[t] * do_
            ddg_ = (1 - g_[t] ** 2) * dg_
            da_ = np.hstack((ddi_.ravel(), ddf_.ravel(), ddo_.ravel(), ddg_.ravel()))
             # backprop through tanh nonlinearity #tanh'(x) = 1-tanh^2(x)
            dLdU_[:, y[t]] += da_
            dLdW_ += np.outer(da_, s[t - 1])
            dsnext = np.dot(self.W_.T, da_)

        dh_minus_one=dsnext
        dc_minus_one=dcnext_

        T1 = len(x)

        for dparam in [dLdV , dLdU_, dLdW_, dLdnew_W]:
            np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients.

        dLdU = np.zeros(self.U.shape)
        dLdW = np.zeros(self.W.shape)

        for t in np.arange(T1)[::-1]:
            # Initial delta calculation
            if t==T1-1:
                dhraw=dh_dot_minus_one[t]+dh_minus_one
            else:
                dhraw=dh_dot_minus_one[t]+dhnext
            dh = (1 - h[t] * h[t]) * dhraw  # backprop through tanh nonlinearity #tanh'(x) = 1-tanh^2(x)
            if t==T1-1:
                dc = dc_minus_one + (1 - np.tanh(c[t]) * np.tanh(c[t])) * dh * o[t]  # backprop through tanh nonlinearity
            else:
                dc = dcnext + (1 - np.tanh(c[t]) * np.tanh(c[t])) * dh * o[t]  # backprop through tanh nonlinearity
            dcnext = dc * f[t]
            di = dc * g[t]
            df = dc * c[t - 1]
            do = dh * np.tanh(c[t])
            dg = dc * i[t]
            ddi = (1 - i[t]) * i[t] * di
            ddf = (1 - f[t]) * f[t] * df
            ddo = (1 - o[t]) * o[t] * do
            ddg = (1 - g[t] ** 2) * dg
            da = np.hstack((ddi.ravel(), ddf.ravel(), ddo.ravel(), ddg.ravel()))
            dLdU[:, x[t]] += da
            dLdW += np.outer(da, h[t - 1])
            dhnext = np.dot(self.W.T, da)

        for dparam in [dLdV, dLdU, dLdW]:
            np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients.

        return [dLdU, dLdW, dLdU_, dLdW_, dLdV, dLdnew_W, dgamma, dbeta]

    def predict(self, x):
        T1 = len(x)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        h = np.zeros((T1 + 1, self.hidden_dim))
        h[-1] = np.zeros(self.hidden_dim)
        c = np.zeros((T1 + 1, self.hidden_dim))
        o = np.zeros((T1, self.hidden_dim))
        i = np.zeros((T1, self.hidden_dim))
        f = np.zeros((T1, self.hidden_dim))
        g = np.zeros((T1, self.hidden_dim))
        # The outputs at each time step. Again, we save them for later.
        # For each time step...
        H = self.hidden_dim
        for t in np.arange(T1):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            temp = self.U[:, x[t]] + self.W.dot(h[t - 1])
            i[t] = sigmoid(temp[:H])
            f[t] = sigmoid(temp[H:2 * H])
            o[t] = sigmoid(temp[2 * H:3 * H])
            g[t] = np.tanh(temp[3 * H:])
            c[t] = f[t] * (c[t - 1]) + i[t] * (g[t])
            h[t] = o[t] * np.tanh(c[t])


        s_pre = h[-2]
        c__pre=c[-2]
        z=0
        pred=[]
        att=[]

        # The outputs at each time step. Again, we save them for later.
        output = np.zeros((self.word_dim2))
        # For each time step...
        step=0
        while z!=1 and step<=10:
            temp_ = self.U_[:, z] + self.W_.dot(s_pre)
            i_ = sigmoid(temp_[:H])
            f_ = sigmoid(temp_[H:2 * H])
            o_ = sigmoid(temp_[2 * H:3 * H])
            g_ = np.tanh(temp_[3 * H:])
            c_ = f_ * (c__pre) + i_ * (g_)
            s = o_[t] * np.tanh(c_)
            e_pre = np.dot(h[:-1], self.new_W)
            e = np.dot(e_pre, s.T)
            alpha = softmax(e)
            a = np.dot(alpha.T, h[:-1])
            result = np.concatenate((a, s), 0)
            output = softmax(self.V.dot(result))
            z = np.argmax(output)
            pred.append(z)
            att.append(alpha)
            s_pre = s
            c__pre=c_
            step+=1


        return pred, att

    def calculate_total_loss(self, x, y):
        L = 0
        # For each sentence...
        for i in np.arange(len(y)):
            output, _, _, _, _, _, _ , _, _, _, _, _, _, _, _, _, _  = self.forward_propagation(x[i], y[i])
            y_=y[i][1:]
            y_.append(1)
            # We only care about our prediction of the "correct" words
            correct_word_predictions = output[np.arange(len(y[i])), y_]
            #print('correct_word_predictions :', correct_word_predictions)
            # Add to the loss based on how off we were
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L


    def calculate_loss(self, x, y):
        # Divide the total loss by the number of training examples
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x, y) / N

    def gradient_check_loss(self, x, y):
        L = 0
        # For each sentence...
        output, _, _, _, _, _, _ , _, _, _, _, _, _, _, _, _  = self.forward_propagation(x, y)
        y_=y[1:]
        y_.append(1)
        # We only care about our prediction of the "correct" words
        correct_word_predictions = output[np.arange(len(y)), y_]
        #print('correct_word_predictions :', correct_word_predictions)
        # Add to the loss based on how off we were
        L += -1 * np.sum(np.log(correct_word_predictions))
        return L

    def gradCheck(self, x, y):
        global Wxh, Whh, Why, bh, by
        num_checks, delta = 10, 1e-3
        dLdU, dLdW, dLdU_, dLdW_, dLdV = self.bptt(x, y)
        for param, dparam, name in zip([self.U, self.W, self.U_, self.W_, self.V], [dLdU, dLdW, dLdU_, dLdW_, dLdV],
                                       ['U', 'W', 'U_', 'W_', 'V']):
            s0 = dparam.shape
            s1 = param.shape
            assert s0 == s1
            print(name)
            for i in np.arange(num_checks):
                ri = int(uniform(0, param.size))
                # evaluate cost at [x + delta] and [x - delta]
                old_val = param.flat[ri]
                param.flat[ri] = old_val + delta
                cg0 = self.gradient_check_loss(x, y)
                param.flat[ri] = old_val - delta
                cg1 = self.gradient_check_loss(x, y)
                param.flat[ri] = old_val  # reset old value for this parameter
                # fetch both numerical and analytic gradient
                grad_analytic = dparam.flat[ri]
                print('cg0 :', cg0)
                print('cg1 :', cg1)
                grad_numerical = (cg0 - cg1) / (2 * delta)
                rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
                print('%f, %f => %e ' % (grad_numerical, grad_analytic, rel_error))
                # rel_error should be on order of 1e-7 or less


    # Performs one step of SGD.
    def numpy_sdg_step(self, x, y, learning_rate):
        # Calculate the gradients
        dLdU, dLdW, dLdU_, dLdW_, dLdV, dLdnew_W, dgamma, dbeta = self.bptt(x, y)
        #print('dLdU :', dLdU)
        #print('dLdW :', dLdW)
        #print('dLdU_ :', dLdU_)
        #print('dLdW_ :', dLdW_)
        #print('dLdV :', dLdV)
        # Change parameters according to gradients and learning rate

        self.mU = self.beta1 * self.mU + (1 - self.beta1) * dLdU
        self.mV = self.beta1 * self.mV + (1 - self.beta1) * dLdV
        self.mW = self.beta1 * self.mW + (1 - self.beta1) * dLdW
        self.mU_ = self.beta1 * self.mU_ + (1 - self.beta1) * dLdU_
        self.mW_ = self.beta1 * self.mW_ + (1 - self.beta1) * dLdW_
        self.mnew_W = self.beta1 * self.mnew_W + (1 - self.beta1) * dLdnew_W


        self.vU = self.beta2 * self.vU + (1 - self.beta2) * (dLdU ** 2)
        self.vV = self.beta2 * self.vV + (1 - self.beta2) * (dLdV ** 2)
        self.vW = self.beta2 * self.vW + (1 - self.beta2) * (dLdW ** 2)
        self.vU_ = self.beta2 * self.vU_ + (1 - self.beta2) * (dLdU_ ** 2)
        self.vW_ = self.beta2 * self.vW_ + (1 - self.beta2) * (dLdW_ ** 2)
        self.vnew_W = self.beta2 * self.vnew_W + (1 - self.beta2) * (dLdnew_W ** 2)

        self.U += -learning_rate * self.mU / (np.sqrt(self.vU) + self.eps)
        self.V += -learning_rate * self.mV / (np.sqrt(self.vV) + self.eps)
        self.W += -learning_rate * self.mW / (np.sqrt(self.vW) + self.eps)
        self.U_ += -learning_rate * self.mU_ / (np.sqrt(self.vU_) + self.eps)
        self.W_ += -learning_rate * self.mW_ / (np.sqrt(self.vW_) + self.eps)
        self.new_W += -learning_rate * self.mnew_W / (np.sqrt(self.vnew_W) + self.eps)

        # gamma, beta를 update한다.
        self.mgamma = self.beta1 * self.mgamma + (1 - self.beta1) * dgamma
        self.mbeta = self.beta1 * self.mbeta + (1 - self.beta1) * dbeta

        self.vgamma = self.beta2 * self.vgamma + (1 - self.beta2) * (dgamma ** 2)
        self.vbeta = self.beta2 * self.vbeta + (1 - self.beta2) * (dbeta ** 2)

        self.gamma += -learning_rate * self.mgamma / (np.sqrt(self.vgamma) + self.eps)
        self.beta += -learning_rate * self.mbeta / (np.sqrt(self.vbeta) + self.eps)




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
model = LSTM_attention_multi(input_lang.n_words, output_lang.n_words)
#model.gradCheck(X_train[0], y_train[0])
losses = model.train_with_sgd(X_train, y_train, nepoch=50, evaluate_loss_after=1)


def predict(input_number):
    x=pairs[:,0][input_number]
    print('input :', x)
    y=[]

    for word in x.split():
        y.append(input_lang.word2index[word])


    #print('y :', y)
    print('result :', pairs[:,1][input_number])
    #print('predict :', model.predict(y))
    print('predict :', [output_lang.index2word[i] for i in model.predict(y)[0]])
    print()


predict(0)
predict(5)
predict(10)
predict(15)
predict(20)