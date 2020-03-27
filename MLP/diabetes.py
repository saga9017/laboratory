import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split



# 당뇨병 데이터 읽어오기
xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=123)





# Placeholders : Shape 주의! 총 8개의 x_data와 1개의 y_data
X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W1 = tf.Variable(tf.random_normal([8, 16]), name='weight')
b1 = tf.Variable(tf.random_normal([16]), name='bias')

W2 = tf.Variable(tf.random_normal([16, 1]), name='weight')
b2 = tf.Variable(tf.random_normal([16]), name='bias')

# Hypothesis
hypothesis = tf.nn.relu(tf.sigmoid(tf.matmul(X, W1) + b1))
hypothesis = tf.sigmoid(tf.matmul(hypothesis, W2) + b2)


# Cost/Loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# 정확도 hypothesis > 0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# 세션 시작
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(20000):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_train, Y: y_train})
        if step % 2000 == 0:
            print('step :', step, 'train cost :', cost_val)
    # 10000 0.480384

# 정확도 77%
    _, _, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_test, Y: y_test})
    print("Test Accuracy: ", a)
# Accuracy: 0.769433
