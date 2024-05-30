import numpy as np

class AutoDiff:
    def __init__(self, value, depend=None, opt=''):
        self.value = value  # 该节点的值
        self.depend = depend  # 生成该节点的两个子节点
        self.opt = opt  # 两个子节点的运算方式
        self.grad = None  # 函数对该节点的梯度

    def add(self, y):
        result = self.value + y.value
        return AutoDiff(value=result, depend=[self, y], opt='add')

    def dot(self,y):
        result = np.dot(self.value, y.value)
        return AutoDiff(value=result, depend=[self, y], opt='dot')

    def multiply(self, y):
        result = self.value * y.value
        return AutoDiff(value=result, depend=[self, y], opt='multiply')

    def subtract(self, y):
        result = self.value - y.value
        return AutoDiff(value=result, depend=[self, y], opt='subtract')

    def divide(self, y):
        result = self.value / y.value
        return AutoDiff(value=result, depend=[self, y], opt='divide')

    def power(self, y):
        result = self.value ** y.value
        return AutoDiff(value=result, depend=[self, y], opt='power')

    def Sigmoid(self):
        s = 1 / (1 + np.exp(-self.value))

        return AutoDiff(value=s, depend=[self], opt='Sigmoid')

    def ReLU(self):
        s = np.maximum(0, self.value)

        return AutoDiff(value=s, depend=[self], opt='ReLU')

    def tanh(self):
        s = np.tanh(self.value)

        return AutoDiff(value=s, depend=[self], opt='tanh')

    def backward(self, backward_grad=None):
        '''反向求导'''
        if backward_grad is None:
            if self.value.ndim == 0:
                self.grad = np.array([1])
            else:
                self.grad = np.ones_like(self.value)
        else:
            if self.grad is None:
                self.grad = backward_grad
            else:
                if self.grad.shape != backward_grad.shape:
                    backward_grad = np.sum(backward_grad, axis=1, keepdims=True)
                self.grad += backward_grad
        if self.opt == 'add':
            self.depend[0].backward(self.grad)
            self.depend[1].backward(self.grad)
        elif self.opt == 'subtract':
            new = self.grad
            self.depend[0].backward(new)
            new = -self.grad
            self.depend[1].backward(new)
        elif self.opt == 'dot':
            new = np.dot(self.grad, self.depend[1].value.T)
            self.depend[0].backward(new)
            new = np.dot(self.depend[0].value.T, self.grad)
            self.depend[1].backward(new)
        elif self.opt == 'multiply':
            new = self.grad * self.depend[1].value
            self.depend[0].backward(new)
            new = self.grad * self.depend[0].value
            self.depend[1].backward(new)
        elif self.opt == 'divide':
            new = self.grad / self.depend[1].value
            self.depend[0].backward(new)
            new = -self.grad * self.depend[0].value / (self.depend[1].value ** 2)
            self.depend[1].backward(new)
        elif self.opt == 'power':
            new = self.grad * self.value ** (self.depend[1].value - 1)
            self.depend[0].backward(new)
            new = self.grad * self.value ** self.depend[1].value
            self.depend[1].backward(new)

        elif self.opt == 'Sigmoid':
            new = self.grad * (1 / (1 + np.exp(-self.depend[0].value))) * (1 - 1 / (1 + np.exp(-self.depend[0].value)))
            self.depend[0].backward(new)
        elif self.opt == 'ReLU':
            new = self.grad * (self.depend[0].value > 0)
            self.depend[0].backward(new)
        elif self.opt == 'tanh':
            new = self.grad * (1 - np.tanh(self.depend[0].value) ** 2)
            self.depend[0].backward(new)

    def loss_compute(self, y):
        loss = -np.mean(y * np.log(self.value) + (1 - y) * np.log(1 - self.value))

        return loss
