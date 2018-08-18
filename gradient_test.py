import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, [None, 2])
y = tf.placeholder(tf.float32, [None, 1])

def _net(net, scope):
    with tf.variable_scope(scope):
        for i in range(2):
            net = tf.layers.dense(net, 64, activation=tf.nn.tanh)
        net = tf.layers.dense(net, 1)
    return net

logits = _net(x, 'ptc1')
u_log_prob = - tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)  # unnormalized log likelihood

ptc1_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ptc1')

grad_ptc1 = tf.gradients(u_log_prob, ptc1_var)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    g = sess.run(grad_ptc1, feed_dict={
        x: np.ones((4, 2)),
        y: np.ones((4, 1))
    })
    print([v.shape for v in g])

