import tensorflow as tf
from tensorflow.contrib.distributions import percentile
import numpy as np


class SVGD(object):
    def __init__(self, grads_list, vars_list, optimizer, learning_rate, median_heuristic=True):
        self.grads_list = grads_list
        self.vars_list = vars_list
        self.optimizer = optimizer
        self.learning_rate =learning_rate
        self.num_particles = len(vars_list)
        self.median_heuristic = median_heuristic
        self.update_op = self.build_optimizer()

    def build_optimizer(self):
        flatgrads_list, flatvars_list = [], []

        for i in range(self.num_particles):
            grads, vars = self.grads_list[i], self.vars_list[i]
            flatgrads, flatvars = self.flatten_grads_and_vars(grads, vars)
            flatgrads_list.append(flatgrads)
            flatvars_list.append(flatvars)

        def SVGD_kernel(flatvars_list):
            # For pairwise distance in a matrix form, I use the following reference:
            #       https://stackoverflow.com/questions/37009647
            #               /compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
            stacked_vars = tf.stack(flatvars_list)
            norm = tf.reduce_sum(stacked_vars*stacked_vars, 1)
            norm = tf.reshape(norm, [-1, 1])
            pairwise_dists = norm - 2 * tf.matmul(stacked_vars, tf.transpose(stacked_vars)) + tf.transpose(norm)

            # For median in TensorFlow, I use the following reference:
            #       https://stackoverflow.com/questions/43824665/tensorflow-median-value
            def _percentile(x, interpolation):
                return percentile(x, 50.0, interpolation=interpolation)

            if self.median_heuristic:
                median = (_percentile(pairwise_dists, 'lower') + _percentile(pairwise_dists, 'higher')) / 2.
                median = tf.cast(median, tf.float32)
                h = tf.sqrt(0.5 * median / tf.log(self.num_particles + 1.))

            if self.num_particles == 1:
                h = 1.

            # kernel computation
            Kxy = tf.exp(- pairwise_dists / h ** 2 / 2)
            dxkxy = - tf.matmul(Kxy, stacked_vars)
            sumkxy = tf.reduce_sum(Kxy, axis=1, keep_dims=True)
            dxkxy = (dxkxy + stacked_vars * sumkxy) / h ** 2

            return Kxy, dxkxy

        # gradients of SVGD
        Kxy, dxkxy = SVGD_kernel(flatvars_list)
        stacked_grads = tf.stack(flatgrads_list)
        stacked_grads = (tf.matmul(Kxy, stacked_grads) + dxkxy) / self.num_particles
        flatgrads_list = tf.unstack(stacked_grads, self.num_particles)

        # make gradients for each particle
        grads_list = []
        for flatgrads, vars in zip(flatgrads_list, self.vars_list):
            start = 0
            grads = []
            for var in vars:
                shape = self.var_shape(var)
                size = int(np.prod(shape))
                grads.append(tf.reshape(flatgrads[start:start + size], shape))
            grads_list.append(grads)

        # optimizer
        update_ops = []
        for grads, vars in zip(grads_list, self.vars_list):
            opt = self.optimizer(learning_rate=self.learning_rate)
            update_ops.append(opt.apply_gradients([(g, v) for g, v in zip(grads, vars)]))
        return tf.group(*update_ops)

    def flatten_grads_and_vars(self, grads, vars):
        """Flatten gradients and variables (from openai/baselines/common/tf_util.py)

        :param grads: list of gradients
        :param vars: list of variables
        :return: two lists of flattened gradients and varaibles
        """
        flatgrads =  tf.concat(axis=0, values=[
            tf.reshape(grad if grad is not None else tf.zeros_like(var), [self.num_elements(var)])
            for (var, grad) in zip(vars, grads)])
        flatvars = tf.concat(axis=0, values=[
            tf.reshape(var, [self.num_elements(var)])
            for var in vars])
        return flatgrads, flatvars

    def num_elements(self, var):
        return int(np.prod(self.var_shape(var)))

    @staticmethod
    def var_shape(var):
        out = var.get_shape().as_list()
        assert all(isinstance(a, int) for a in out), \
            'shape function assumes that shape is fully known'
        return out


# data generation
mean0 = np.array([-1, -1])
std0 = np.array([1, 1])
mean1 = np.array([1, 1])
std1 = np.array([1, 1])
x0 = np.tile(mean0, (100, 1)) + std0 * np.random.randn(100, 2)
x1 = np.tile(mean1, (100, 1)) + std1 * np.random.randn(100, 2)
y0 = np.zeros((x0.shape[0], 1))
y1 = np.ones((x1.shape[0], 1))

x = np.concatenate([x0, x1], axis=0)
y = np.concatenate([y0, y1], axis=0)
D = np.hstack([x, y])
np.random.shuffle(D)
x = np.array(D[:, 0:2], dtype=np.float32)
y = np.array(D[:, 2:], dtype=np.float32)

x_ = tf.placeholder(tf.float32, [None, 2])
y_ = tf.placeholder(tf.float32, [None, 1])

# number of particles
num_particles = 50

def network(inputs, labels, scope):
    net = inputs
    with tf.variable_scope(scope):
        for _ in range(2):
            net = tf.layers.dense(net, 100, activation=tf.nn.tanh)
        logits = tf.layers.dense(net, 1)
        # Based on the model assumption
        #       p(w, D) := p(w) \prod_{i=1}^N p(x_i) p(0|x_i,w)^{1-y_i} p(1|x_i,w)^{y_i},
        # the log likelihood is equal to the negative cross entropy up to unknown normalization constant.
        log_likelihood = - tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        # uniform prior assumption
        gradients = tf.gradients(log_likelihood, variables)

    return gradients, variables

grads_list = []
vars_list = []
for i in range(num_particles):
    grads, vars = network(x_, y_, 'p{}'.format(i))
    grads_list.append(grads)
    vars_list.append(vars)

SVGDoptimizer = SVGD(grads_list=grads_list,
                     vars_list=vars_list,
                     optimizer=tf.train.AdamOptimizer,
                     learning_rate=0.001)

import time
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    start_time = time.time()
    for i in range(100):
        print(i)
        sess.run(SVGDoptimizer.update_op, feed_dict={x_: x, y_: y})
        end_time = time.time()
        print(end_time - start_time)
        start_time = end_time

