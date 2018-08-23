import numpy as np
from svgd import SVGD
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import sys; sys.path.insert(0, '..')
from utils import Time


class GaussianMixtureModel(object):
    def __init__(self):
        pass

    def dlnprob(self, x):
        # Derivative of Gaussian mixture was calculated by Wolfram alpha.
        # D[Log[(Exp[-(x+2)^2 / 2] / 3 + 2 / 3 Exp[-(x-2)^2 / 2]) / Sqrt[2 Pi]], x]
        return - x - 4 / (2 * np.exp(4 * x) + 1) + 2


# hyper-parameters
num_particles = 100  # number of ensembles (SVGD particles)
num_iterations = 2000  # number of training iterations
learning_rate = 0.01
seed = 0

# random seeds
np.random.seed(seed)

model = GaussianMixtureModel()

with Time("Get initial particles"):
    initial_xs = np.array(np.random.normal(-10, 1, (100, 1)), dtype=np.float32)
with Time("training & Get last particles"):
    final_xs = SVGD().update(initial_xs, model.dlnprob, n_iter=num_iterations, stepsize=learning_rate)
initial_xs, final_xs = initial_xs.reshape(-1), final_xs.reshape(-1)


def plot():
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    x_grid = np.linspace(-15, 15, 200)

    initial_density = gaussian_kde(initial_xs)
    ax.plot(x_grid, initial_density(x_grid), color='green', label='0th iteration')
    ax.scatter(initial_xs, np.zeros_like(initial_xs), color='green')

    final_density = gaussian_kde(final_xs)
    ax.plot(x_grid, final_density(x_grid), color='red', label='{}th iteration'.format(num_iterations))
    ax.scatter(final_xs, np.zeros_like(final_xs), color='red')

    def log_normal(x, m, s):
        return - (x - m) ** 2 / 2. / s ** 2 - np.log(s) - 0.5 * np.log(2. * np.pi)
    target_density = np.exp(log_normal(x_grid, -2., 1.)) / 3 + np.exp(log_normal(x_grid, 2., 1.)) * 2 / 3
    ax.plot(x_grid, target_density, 'r--', label='target density')

    ax.set_xlim([-15, 15])
    ax.set_ylim([0, 0.4])
    ax.legend()
    plt.show()


plot()
