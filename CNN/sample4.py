import tensorflow as tf
import numpy as np

max_len=4
vocabulary_size=60


input=np.random.randn(50, max_len)
print(input.shape)
embedding_matrix=tf.Variable(tf.random_normal([vocabulary_size, 300], stddev=0.01))

X = tf.placeholder(tf.int32, [None, max_len])
#X_float=tf.map_fn(lambda x: tf.IndexedSlices(embedding_matrix, x), X)
X_float=tf.map_fn(lambda x: tf.nn.embedding_lookup(embedding_matrix, x), X)

#####################################################################################33
sess = tf.Session()
sess.run(tf.global_variables_initializer())

feed_dict = {X: input}
c= sess.run(X_float, feed_dict=feed_dict)

print(len(c))