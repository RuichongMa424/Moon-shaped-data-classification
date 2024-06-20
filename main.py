from model import model
from auto_diff import AutoDiff


from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt


def load_dataset(plot=False):
    np.random.seed(42)
    train_X, train_Y = make_moons(n_samples=600, noise=.3)
    if plot:
        plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap='summer', edgecolors='k');
        plt.show()
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))

    return train_X, train_Y

def split_dataset(x, y, test_size=0.1):

    # 计算测试集的样本数
    num_test_samples = int(x.shape[1] * test_size)

    # 划分训练集和测试集
    X_train = x[:, :-num_test_samples]
    y_train = y[:, :-num_test_samples]
    X_test = x[:, -num_test_samples:]
    y_test = y[:, -num_test_samples:]

    return X_train, X_test, y_train, y_test


def predict(x, p):
    m = x.shape[1]
    prediction = np.zeros((1, m), dtype=int)

    x = AutoDiff(x)
    l1 = p.w1.dot(x).add(p.b1).ReLU()
    l2 = p.w2.dot(l1).add(p.b2).ReLU()
    l3 = p.w3.dot(l2).add(p.b3).Sigmoid()

    prediction = (l3.value > 0.5)
    return prediction

def plot_result(parameters, x, y):
    #设置最小值和最大值
    x_min, x_max = x[0, :].min() - 1, x[0, :].max() + 1
    y_min, y_max = x[1, :].min() - 1, x[1, :].max() + 1
    h = 0.01
    Cx, Cy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    #预测函数值
    Z = predict(np.c_[Cx.ravel(), Cy.ravel()].T, parameters)
    Z = Z.reshape(Cx.shape)
    #绘制轮廓和数据点
    plt.contourf(Cx, Cy, Z, cmap='jet', alpha=0.5)
    plt.title('Decision Boundary')
    plt.scatter(x[0, :], x[1, :], c=np.squeeze(y), cmap='summer', edgecolors='k')
    plt.show()

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
    Epoch = 600
    learning_rate = 0.1
    hidden_size = 8
    input_size = 2
    output_size = 1
    loss = []
    Accuracy = []

    ##读取数据集代码
    train_X, train_Y = load_dataset(plot=True)
    train_x,test_x, train_y, test_y = split_dataset(train_X, train_Y, test_size=0.1)

    #初始化
    parameters = Parameters(input_size, hidden_size, output_size)

    #训练
    for i in range(Epoch):
        parameters = model(train_x, train_y, lr=learning_rate, p=parameters)
        predictions = predict(test_x, parameters)
        accuracy = float((np.dot(test_y, predictions.T) + np.dot(1 - test_y, 1 - predictions.T)) / float(test_y.size) * 100)
        if (i+1) % 10 == 0:
            print("第 ", i+1, "轮" "，准确率：%.1f%%"% accuracy,"%，loss：" + str(parameters.loss))
        loss.append(parameters.loss)
        Accuracy.append(accuracy)
        if (i+1) % 100 == 0:
            plot_result(parameters, train_X, train_Y)

    # 绘制损失函数曲线
    plt.plot(np.array(loss))
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    # 绘制准确率曲线
    plt.plot(np.array(Accuracy))
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()


    print('最终准确率: %.1f%%' % (Accuracy[-1]))
    return parameters

if __name__ == "__main__":
    main()
