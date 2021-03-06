# TensorFlow Implementation of Stein Variational Gradient Descent (SVGD)

## References
-   [Paper in NIPS 2016](https://arxiv.org/abs/1608.04471)
-   [Authors' code](https://github.com/DartML/Stein-Variational-Gradient-Descent)
-   Some TensorFlow utilities from [OpenAI Baselines](https://github.com/openai/baselines).

## Usages
1. Define network, and get gradients and variables, e.g.,
```python
def network():
    '''
    Define target density and return gradients and variables. 
    '''
    return gradients, variables
```

2. Define gradient descent optimizer, e.g.,
```python
def make_gradient_optimizer():
    return tf.train.GradientDescentOptimizer(learning_rate=0.01)
```

3. Build multiple networks (particles) using `network()` and 
    take all those gradients and variables in `grads_list` and `vars_list`.
    
4. Make SVGD optimizer, e.g., 
```python
optimizer = SVGD(grads_list, vars_list, make_gradient_optimizer)
```

5. In the training phase, `optimizer.update_op` will do single SVGD update, e.g.,
```python
sess = tf.Session()
sess.run(optimizer.update_op, feed_dict={X: x, Y: y})
```


## Examples
### 1D Gaussian mixture
-   The goal of this problem is to match the target density p(x)
    (mixture of two Gaussians)
    by moving the particles initially sampled from other distributions q(x).
    For details, I recommend you to see the experiment section in the authors' paper. 
    
-   I got the following result:

    <p float="left" align="center">
      <img src="/results/1_gaussian_mixture/gmm_result.gif" width="400" />
    </p>
    
-   **NOTE THAT** I compared my implementation with that of authors
    and checked the results are the same.
 

### Bayesian Binary Classification
-   In this example, we want to classify binary data by using multiple neural classifier. 
    I checked how SVGD differs from ensemble method in this example.
    I made a [pdf file](./derivations/bayesian_classification.pdf) for detailed mathematical derivations.

-   I got the following results:

    <p float="left" align="center">
      <img src="/results/2_bayesian_classification/predictive_ensemble_20.png" width="350" />
      <img src="/results/2_bayesian_classification/predictive_svgd_20.png" width="350" />
    </p>

    -   Thus, ensemble methods make particles to *strongly* classify samples,
        where as SVGD leads to draw the particles that characterize the posterior distribution.
