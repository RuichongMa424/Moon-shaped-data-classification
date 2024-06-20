
from auto_diff import AutoDiff


import numpy as np
import math

def model(x,y,lr,p):

    # 将输入建模为AutoDiff类
    x = AutoDiff(x)

    # 前向传播
    l1 = p.w1.dot(x).add(p.b1).ReLU()
    l2 = p.w2.dot(l1).add(p.b2).ReLU()
    l3 = p.w3.dot(l2).add(p.b3).Sigmoid()

    # 计算loss
    p.loss = AutoDiff.loss_compute(l3, y)

    # 反向传播
    m = x.value.shape[1]
    l3.backward(backward_grad = 1./m * (l3.value - y))

    p.b1.grad = np.sum(l1.grad, axis=1, keepdims=True)
    p.b2.grad = np.sum(l2.grad, axis=1, keepdims=True)
    p.b3.grad = np.sum(l3.grad, axis=1, keepdims=True)

    #使用SGD优化
    p.w1.value -= lr * p.w1.grad
    p.b1.value -= lr * p.b1.grad
    p.w2.value -= lr * p.w2.grad
    p.b2.value -= lr * p.b2.grad
    p.w3.value -= lr * p.w3.grad
    p.b3.value -= lr * p.b3.grad

    #清空梯度
    p.w1.grad = None
    p.w2.grad = None
    p.w3.grad = None

    return p