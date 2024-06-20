import numpy as np

class AutoDiff:
    # 类初始化，建立一个AutoDiff类节点，其包括这个节点的值、子节点、子节点运算方式以及此节点的梯度
    def __init__(self, value, depend=None, opt=''):
        self.value = value
        self.depend = depend
        self.opt = opt
        self.grad = None

    # 前向传播可能需要的数学操作，包括加、点乘、乘、减、微分、乘方
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

    # 前向传播可能用到的激活函数，包括sigmoid、relu、tanh、softmax
    def Sigmoid(self):
        s = 1 / (1 + np.exp(-self.value))

        return AutoDiff(value=s, depend=[self], opt='Sigmoid')

    def ReLU(self):
        s = np.maximum(0, self.value)

        return AutoDiff(value=s, depend=[self], opt='ReLU')

    def tanh(self):
        s = np.tanh(self.value)

        return AutoDiff(value=s, depend=[self], opt='tanh')

    def softmax(self):
        s = np.exp(self.value) / np.sum(np.exp(self.value))

        return AutoDiff(value=s, depend=[self], opt='softmax')

    # 反向传播模块
    def backward(self, backward_grad=None):
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

        # 反向传播节点梯度计算，根据每一个节点的运算方式，进行梯度的计算，从根节点遍历整个静态图
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
        elif self.opt == 'softmax':
            new = self.grad
            self.depend[0].backward(new)

    # 计算loss
    def loss_compute(self, y):
        loss = -np.mean(y * np.log(self.value) + (1 - y) * np.log(1 - self.value))

        return loss
