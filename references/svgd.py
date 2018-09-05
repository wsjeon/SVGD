# Reference:
# https://github.com/DartML/Stein-Variational-Gradient-Descent/svgd.py
import numpy as np
from scipy.spatial.distance import pdist, squareform


class SVGD():

    def __init__(self):
        pass

    @staticmethod
    def svgd_kernel(theta, h=-1):
        sq_dist = pdist(theta)
        pairwise_dists = squareform(sq_dist) ** 2
        if h < 0:  # if h < 0, using median trick
            h = np.median(pairwise_dists)
            h = np.sqrt(0.5 * h / np.log(theta.shape[0] + 1))

        # compute the rbf kernel
        Kxy = np.exp(-pairwise_dists / h ** 2 / 2)
        dxkxy = -np.matmul(Kxy, theta)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(theta.shape[1]):
            dxkxy[:, i] = dxkxy[:, i] + np.multiply(theta[:, i], sumkxy)
        dxkxy = np.divide(dxkxy, np.power(h, 2))
        return (Kxy, dxkxy)

    def update(self, x0, lnprob, n_iter=1000, stepsize=1e-3, bandwidth=-1, alpha=0.9, debug=False, optimizer='adagrad'):
        # Check input
        if x0 is None or lnprob is None:
            raise ValueError('x0 or lnprob cannot be None!')

        theta = np.copy(x0)

        # adagrad with momentum
        fudge_factor = 1e-6
        historical_grad = 0
        for iter in range(n_iter):
            if debug and (iter + 1) % 1000 == 0:
                print
                'iter ' + str(iter + 1)

            lnpgrad = lnprob(theta)
            # calculating the kernel matrix
            kxy, dxkxy = self.svgd_kernel(theta, h=-1)
            grad_theta = (np.matmul(kxy, lnpgrad) + dxkxy) / x0.shape[0]

            if optimizer == 'adagrad':
                # adagrad
                if iter == 0:
                    historical_grad = historical_grad + grad_theta ** 2
                else:
                    historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta ** 2)
                adj_grad = np.divide(grad_theta, fudge_factor + np.sqrt(historical_grad))
                theta = theta + stepsize * adj_grad
            elif optimizer == 'sgd':
                theta = theta + stepsize * grad_theta
            else:
                raise NotImplementedError

        return theta
