import tensorflow as tf
import numpy as np
from references.svgd import SVGD as SVGD0
from optimizer import SVGD as SVGD1
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import sys; sys.path.insert(0, '..')
from utils import Time


class GaussianMixtureModel(object):
    def __init__(self):
        pass

    def dlnprob(self, x):
        # D[Log[(Exp[-(x+2)^2 / 2] / 2 + Exp[-(x-2)^2 / 2] / 2) / Sqrt[2 Pi]], x]
        return 2 * np.tanh(2 * x) - x


if __name__ == '__main__':
    # hyper-parameters
    num_particles = 100  # number of ensembles (SVGD particles)
    num_iterations = 3000  # number of training iterations
    learning_rate = 0.05
    seed = 0
    algorithm = 'svgd'

    # random seeds
    np.random.seed(seed)
    tf.set_random_seed(seed)

    model = GaussianMixtureModel()

    with Time("Get initial particles"):
        initial_xs = np.array(np.random.normal(-10, 1, (100, 1)), dtype=np.float32)
        print(np.linalg.norm(initial_xs))
    with Time("training & Get last particles"):
        final_xs = SVGD0().update(initial_xs, model.dlnprob, n_iter=num_iterations, stepsize=learning_rate)
    initial_xs, final_xs = initial_xs.reshape(-1), final_xs.reshape(-1)

    # plot
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    x_grid = np.linspace(-15, 15, 200)

    initial_density = gaussian_kde(initial_xs)
    ax.plot(x_grid, initial_density(x_grid), color='green', label='0th iteration')
    ax.scatter(initial_xs, np.zeros_like(initial_xs), color='green')

    final_density = gaussian_kde(final_xs)
    ax.plot(x_grid, final_density(x_grid), color='red', label='{}th iteration'.format(num_iterations), linewidth=5)
    ax.scatter(final_xs, np.zeros_like(final_xs), color='red')

    def log_normal(x, m, s):
        return - (x - m) ** 2 / 2. / s ** 2 - np.log(s) - 0.5 * np.log(2. * np.pi)
    target_density = np.exp(log_normal(x_grid, -2., 1.)) / 2 + np.exp(log_normal(x_grid, 2., 1.)) / 2
    ax.plot(x_grid, target_density, 'r--', label='target density')

    ax.set_xlim([-15, 15])
    ax.set_ylim([0, 0.4])
    ax.legend()

    def network(scope):
        def tf_log_normal(x, m, s):
            return - (x - m) ** 2 / 2. / s ** 2 - tf.log(s) - 0.5 * tf.log(2. * np.pi)

        with tf.variable_scope(scope):
            x = tf.Variable(initial_xs[eval(scope[1:])])
            log_prob0, log_prob1 = tf_log_normal(x, -2., 1.), tf_log_normal(x, 2., 1.)
            # log of target distribution p(x)
            # log_p = tf.reduce_logsumexp(tf.stack([log_prob0, log_prob1]), axis=0) - tf.log(2.)
            l1, l2 = - (x + 2) ** 2 / 2, - (x - 2) ** 2 / 2
            log_p = tf.reduce_logsumexp(tf.stack([l1, l2], axis=0)) - tf.log(np.sqrt(8.*np.pi, dtype=np.float32))
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
            gradients = tf.gradients(log_p, variables)
        return gradients, variables


    def make_gradient_optimizer():
        # return AdagradOptimizer(learning_rate=learning_rate)
        # return tf.train.AdamOptimizer(learning_rate=learning_rate)
        return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)


    with Time("graph construction"):
        grads_list, vars_list = [], []
        for i in range(num_particles):
            grads, vars = network('p{}'.format(i))

#            # grads replacement
#            new_grads = []
#            for g, v in zip(grads, vars):
#                new_grads.append(2 * tf.tanh(2 * v) - v)
#            grads_list.append(new_grads)
            grads_list.append(grads)
            vars_list.append(vars)

        if algorithm == 'svgd':
            optimizer = SVGD1(grads_list=grads_list,
                             vars_list=vars_list,
                             make_gradient_optimizer=make_gradient_optimizer)
        elif algorithm == 'ensemble':
            optimizer = Ensemble(grads_list=grads_list,
                                 vars_list=vars_list,
                                 make_gradient_optimizer=make_gradient_optimizer)
        else:
            raise NotImplementedError

        get_particles_op = tf.trainable_variables()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        with Time("Get initial particles"):
            initial_xs = sess.run(get_particles_op)
            print(np.linalg.norm(initial_xs))
        with Time("training"):
            for _ in range(num_iterations):
                sess.run(optimizer.update_op)
        with Time("Get last particles"):
            final_xs = sess.run(get_particles_op)


        final_density = gaussian_kde(final_xs)
        ax.plot(x_grid, final_density(x_grid), color='blue', label='{}th iteration (TF)'.format(num_iterations))
        ax.scatter(final_xs, np.zeros_like(final_xs), color='blue')

        plt.show()
