from model import model, random_mini_batches
from auto_diff import AutoDiff
from plot import plot_decision_boundary

from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt


def load_dataset(is_plot=True):
    np.random.seed(42)
    train_X, train_Y = make_moons(n_samples=600, noise=.2)
    if is_plot:
        plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
        plt.show()
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))

    return train_X, train_Y

def split_dataset(x, y, test_size=0.1):
    # 检查输入数据的形状是否正确
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("输入数据的形状不正确。x 应该是一个 2D ndarray, y 应该是一个 1D ndarray,且它们的第二个维度大小应该相同。")

    # 计算测试集的样本数
    num_test_samples = int(x.shape[1] * test_size)

    # 划分训练集和测试集
    X_train = x[:, :-num_test_samples]
    y_train = y[:, :-num_test_samples]
    X_test = x[:, -num_test_samples:]
    y_test = y[:, -num_test_samples:]

    return X_train, X_test, y_train, y_test

def predict_dec(x, p):
    """
    Used for plotting decision boundary.

    Arguments:
    parameters -- python dictionary containing your parameters
    X -- input data of size (m, K)

    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """

    # Predict using forward propagation and a classification threshold of 0.5

    x = AutoDiff(x)
    l1 = p.w1.dot(x).add(p.b1).ReLU()
    l2 = p.w2.dot(l1).add(p.b2).ReLU()
    l3 = p.w3.dot(l2).add(p.b3).Sigmoid()

    predictions = (l3.value > 0.5)
    return predictions

def predict(x, y, p):
    m = x.shape[1]
    prediction = np.zeros((1, m), dtype=int)

    ''''# Forward propagation
    z1 = add(multiply(x, p.w1), p.b1)
    a1 = ReLU(z1)
    z2 = add(multiply(a1, p.w2), p.b2)
    a2 = ReLU(z2)
    z3 = add(multiply(a2, p.w3), p.b3)
    a3 = Sigmoid(z3)'''

    x = AutoDiff(x)
    l1 = p.w1.dot(x).add(p.b1).ReLU()
    l2 = p.w2.dot(l1).add(p.b2).ReLU()
    l3 = p.w3.dot(l2).add(p.b3).Sigmoid()

    # convert probas to 0/1 predictions
    for i in range(0, l3.value.shape[1]):
        if l3.value[0, i] > 0.5:
            prediction[0, i] = 1
        else:
            prediction[0, i] = 0

    #print("Accuracy: " + str(np.mean((prediction[0, :] == y[0, :]))))
    return prediction

class Parameters:
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = AutoDiff(np.random.randn(hidden_size, input_size))
        self.b1 = AutoDiff(np.zeros((hidden_size, 1)))
        self.w2 = AutoDiff(np.random.randn(hidden_size, hidden_size))
        self.b2 = AutoDiff(np.zeros((hidden_size, 1)))
        self.w3 = AutoDiff(np.random.randn(output_size,hidden_size,))
        self.b3 = AutoDiff(np.zeros((1, 1)))
        self.grads = None

def main():
    # 设置参数
    seed = 42
    Epoch = 600
    optimizer = 'sgd'
    learning_rate = 0.1
    hidden_size = 8
    input_size = 2
    output_size = 1
    loss = []
    Accuracy = []

    ##读取数据集代码
    train_X, train_Y = load_dataset(is_plot=True)
    Xtrain, Xtest, Ytrain, Ytest = split_dataset(train_X, train_Y, test_size=0.1)

    #初始化
    parameters = Parameters(input_size, hidden_size, output_size)

    for i in range(Epoch):
        minibatches = random_mini_batches(Xtrain, Ytrain, 64, seed)
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            parameters = model(minibatch_X, minibatch_Y, optimizer=optimizer, lr=learning_rate, p=parameters)
            predictions = predict(Xtest, Ytest, parameters)
            accuracy = float((np.dot(Ytest, predictions.T) + np.dot(1 - Ytest, 1 - predictions.T)) / float(Ytest.size) * 100)
        if (i+1) % 10 == 0:
            print("第 ", i+1, "轮，loss：" + str(parameters.loss), "，准确率："+ str(accuracy),"%")
        loss.append(parameters.loss)
        Accuracy.append(accuracy)
        if (i+1) % 100 == 0:
            plot_decision_boundary(lambda x: predict_dec(x.T, parameters), train_X, train_Y)


    # 绘制损失函数曲线
    plt.plot(np.array(loss))
    plt.title('Loss function plot with ' + optimizer)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    # 绘制准确率曲线
    plt.plot(np.array(Accuracy))
    plt.title('Accuracy plot with ' + optimizer)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()


    print('最终准确率: %.1f%%' % (Accuracy[-1]))
    return parameters

if __name__ == "__main__":
    main()
