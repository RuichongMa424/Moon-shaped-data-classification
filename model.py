
from auto_diff import AutoDiff


import numpy as np
import math

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[1]                  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size : (k+1)*mini_batch_size]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, num_complete_minibatches*mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches*mini_batch_size : m]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def model(x,y,optimizer,lr,p):
    '''#前向传播
    z1 = add(multiply(p.w1, x), p.b1)
    a1 = ReLU(z1)
    z2 = add(multiply(p.w2, a1), p.b2)
    a2 = ReLU(z2)
    z3 = add(multiply(p.w3, a2), p.b3)
    a3 = Sigmoid(z3)'''


    '''# 反向传播
    m = x.shape[1]
    da3 = (a3 - y) * 1/m
    dz3 = da3
    dw3 = multiply(dz3, a2.T)
    db3 = dz3

    da2 = multiply(p.w3.T, dz3)
    dz2 = multiply(da2, ReLU(z2))
    dw2 = multiply(a1.T, dz2)
    db2 = dz2

    da1 = multiply(p.w2.T, dz2)
    dz1 = multiply(da1, ReLU(z1))
    dw1 = multiply(x.T, dz1)
    db1 = dz1'''

    x = AutoDiff(x)

    l1 = p.w1.dot(x).add(p.b1).ReLU()
    l2 = p.w2.dot(l1).add(p.b2).ReLU()
    l3 = p.w3.dot(l2).add(p.b3).Sigmoid()



    p.loss = AutoDiff.loss_compute(l3, y)
    m = x.value.shape[1]
    l3.backward(backward_grad = 1./m * (l3.value - y))

    p.b1.grad = np.sum(l1.grad, axis=1, keepdims=True)
    p.b2.grad = np.sum(l2.grad, axis=1, keepdims=True)
    p.b3.grad = np.sum(l3.grad, axis=1, keepdims=True)


    if optimizer == 'sgd':
        p.w1.value -= lr * p.w1.grad
        p.b1.value -= lr * p.b1.grad
        p.w2.value -= lr * p.w2.grad
        p.b2.value -= lr * p.b2.grad
        p.w3.value -= lr * p.w3.grad
        p.b3.value -= lr * p.b3.grad

    p.w1.grad = None
    p.w2.grad = None
    p.w3.grad = None

    return p