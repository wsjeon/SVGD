import tensorflow as tf
import numpy as np
import sys; sys.path.insert(0, '..')
from references.svgd import SVGD as SVGD0
from optimizer import SVGD as SVGD1
from utils import Time


if __name__ == '__main__':
    # hyper-parameters
    num_particles = 100  # number of ensembles (SVGD particles)
    seed = 0

    # random seeds
    np.random.seed(seed)

    with Time("Get initial particles"):
        initial_xs = np.array(np.random.normal(-10, 1, (300, 3)), dtype=np.float32)
    if len(initial_xs.shape) == 1:
        initial_xs = initial_xs.reshape(-1, 1)
    Kxy0, dxkxy0 = SVGD0.svgd_kernel(theta=initial_xs)

    with tf.Session() as sess:
        initial_xs_list = []
        for x in initial_xs.tolist():
            initial_xs_list.append(tf.constant(x, dtype=tf.float32))
        Kxy1, dxkxy1 = sess.run(SVGD1.svgd_kernel(initial_xs_list))

        print(np.linalg.norm(Kxy0))
        print(np.linalg.norm(dxkxy0))
        print(np.linalg.norm(Kxy1))
        print(np.linalg.norm(dxkxy1))



