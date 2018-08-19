import tensorflow as tf
import numpy as np

np.random.seed(0)
x = np.random.normal(3.0, .1, 100)
x = np.array([[8, 8, 4], [3, 2, 1], [0, 0, 0]], dtype=np.float64) # 2
x = np.array([[8, 8, 4, 3], [2, 1, 0, 0]], dtype=np.float64) # 2.5

x = tf.constant(x)

median = tf.contrib.distributions.percentile(x, 50.0, interpolation='lower')
median += tf.contrib.distributions.percentile(x, 50.0, interpolation='higher')
median /= 2.

with tf.Session() as sess:
    print(sess.run(median))
