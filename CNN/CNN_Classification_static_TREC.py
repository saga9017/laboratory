import tensorflow as tf
import random
import numpy as np
import gensim
import re


tf.logging.set_verbosity(tf.logging.ERROR)
# import matplotlib.pyplot as plt

embedding_size=300
#train set과 test set 만들기
################################################################################################
input1=[]
input2=[]

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

TREC_train=[]
TREC_test=[]
with open('CNN Data/TREC/TREC.train.all', 'r', encoding='latin-1') as f:
    for row in f.readlines():
        TREC_train.append([clean_str(row[1:], TREC=True), int(row[0])])

# sentences of test data
with open('CNN Data/TREC/TREC.test.all', 'r', encoding='latin-1') as f:
    for row in f.readlines():
        TREC_test.append([clean_str(row[1:], TREC=True), int(row[0])])


def onehot(index):
    output=[0]*index+[1]+[0]*(6-index-1)

    return output


train_x=[]
train_y=[]
for i in TREC_train:
    train_x.append(i[0])
    train_y.append(onehot(i[1]))

test_x = []
test_y = []
for i in TREC_test:
    test_x.append(i[0])
    test_y.append(onehot(i[1]))

#embedding_matrix 만들기
##########################################################################################
max_len=0
for sentence in train_x+test_x:
    if max_len<=len(sentence.split()):
        max_len=len(sentence.split())

index_of_word={}
index_of_word[' ']=0
index=1
max_len=0
for sentence in train_x+test_x:
    if max_len<=len(sentence.split()):
        max_len=len(sentence.split())
    for word in sentence.split():
        if word not in index_of_word.keys():
            index_of_word[word]=index
            index+=1

print('index_of_word :', index_of_word)
#embedding_matrix, index_of_word=load_vector()
model = gensim.models.KeyedVectors.load_word2vec_format('CNN Data/GoogleNews-vectors-negative300.bin.gz', binary=True)

embedding_matrix = np.zeros((len(index_of_word), embedding_size))
for key in index_of_word.keys():
    if key not in model.wv:
        embedding_vector=np.random.uniform(-0.001, 0.001, embedding_size)
    else:
        embedding_vector = model.wv[key]
    embedding_matrix[index_of_word[key]] = embedding_vector

print('max_len :',max_len)
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_len)
vocabulary_size=len(index_of_word)


#print('embedding_matrix :', embedding_matrix.shape)

#train_x, text_x를 수치화 하기
##########################################################################################
for index, sentence in enumerate(train_x):
    temp=[]
    for word in sentence.split():
        temp.append(embedding_matrix[index_of_word[word]])
    #print(np.array(temp).shape)
    padding=np.zeros((max_len-len(temp), embedding_size))
    #print(padding.shape)
    train_x[index]=np.concatenate((np.array(temp), padding))

for index, sentence in enumerate(test_x):
    temp=[]
    for word in sentence.split():
        temp.append(embedding_matrix[index_of_word[word]])
    #print(np.array(temp).shape)
    padding=np.zeros((max_len-len(temp), embedding_size))
    #print(padding.shape)
    test_x[index]=np.concatenate((np.array(temp), padding))

random_index=random.sample(range(0, len(train_x)), len(train_x))
train_x=np.array(train_x)[random_index]
train_y=np.array(train_y)[random_index]


##########################################################################################



tf.set_random_seed(777)  # reproducibility

# hyper parameters
learning_rate = 0.001
training_epochs = 10
batch_size = 50

# input place holders
X = tf.placeholder(tf.float32, [None, max_len, embedding_size])
X_img = tf.reshape(X, [-1, max_len, embedding_size, 1])   # img 28x28x1 (black/white)
Y = tf.placeholder(tf.float32, [None, 6])
keep_prob = tf.placeholder(tf.float32)

# L1 ImgIn shape=(?, 28, 28, 1)
W1 = tf.get_variable("W1", shape=[3, embedding_size, 1, 100],
                     initializer=tf.contrib.layers.xavier_initializer())
#W1 = tf.Variable(tf.random_normal([3, embedding_size, 1, 100], stddev=0.01))
#    Conv     -> (?, 51, 1, 32)
#    Pool     -> (?, 1, 1, 32)
L1_1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='VALID')
L1_2 = tf.nn.relu(L1_1)
L1_3 = tf.nn.max_pool(L1_2, ksize=[1, max_len-2, 1, 1],
                    strides=[1, max_len-2, 1, 1], padding='VALID')
L1_4 = tf.nn.dropout(L1_3, keep_prob)

L1_flat = tf.reshape(L1_4, [-1, 100])

# L2 ImgIn shape=(?, 28, 28, 1)
W2 = tf.get_variable("W2", shape=[4, embedding_size, 1, 100],
                     initializer=tf.contrib.layers.xavier_initializer())
#W2 = tf.Variable(tf.random_normal([4, embedding_size, 1, 100], stddev=0.01))
#    Conv     -> (?, 51, 1, 32)
#    Pool     -> (?, 1, 1, 32)
L2_1 = tf.nn.conv2d(X_img, W2, strides=[1, 1, 1, 1], padding='VALID')
L2_2 = tf.nn.relu(L2_1)
L2_3 = tf.nn.max_pool(L2_2, ksize=[1, max_len-3, 1, 1],
                    strides=[1, max_len-3, 1, 1], padding='VALID')
L2_4 = tf.nn.dropout(L2_3, keep_prob)

L2_flat = tf.reshape(L2_4, [-1, 100])


# L3 ImgIn shape=(?, 28, 28, 1)
W3 = tf.get_variable("W3", shape=[5, embedding_size, 1, 100],
                     initializer=tf.contrib.layers.xavier_initializer())
#W3 = tf.Variable(tf.random_normal([5, embedding_size, 1, 100], stddev=0.01))
#    Conv     -> (?, 51, 1, 32)
#    Pool     -> (?, 1, 1, 32)
L3_1 = tf.nn.conv2d(X_img, W3, strides=[1, 1, 1, 1], padding='VALID')
L3_2 = tf.nn.relu(L3_1)
L3_3 = tf.nn.max_pool(L3_2, ksize=[1, max_len-4, 1, 1],
                    strides=[1, max_len-4, 1, 1], padding='VALID')
L3_4 = tf.nn.dropout(L3_3, keep_prob)

L3_flat = tf.reshape(L3_4, [-1, 100])



# Final FC 7x7x64 inputs -> 10 outputs
W4 = tf.get_variable("W4", shape=[100*3, 6],
                     initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([6]))
logits = tf.matmul(tf.concat([L1_flat ,L2_flat ,L3_flat], 1), W4) + b

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# train my model
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(len(train_x)/ batch_size)
    #print('total_batch :', total_batch)
    start=0
    for i in range(total_batch):
        #print('start :', start)
        batch_xs, batch_ys = train_x[start:start+batch_size], train_y[start:start+batch_size]
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob:0.5}
        c, _= sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
        start+=batch_size


    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), end='  ')
    print('Accuracy:', sess.run(accuracy, feed_dict={
        X: test_x, Y: test_y, keep_prob: 1}))

print('Learning Finished!')




# Get one and predict
r = random.randint(0, len(test_y) - 1)
print("Label: ", sess.run(tf.argmax(test_y[r:r + 1], 1)))
print("Prediction: ", sess.run(
    tf.argmax(logits, 1), feed_dict={X: test_x[r:r + 1], keep_prob:1}))


