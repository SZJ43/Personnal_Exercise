import math
import random
import numpy as np
import sys

np.seterr(all='ignore')


# sigmoid transfer function
# IMPORTANT: when using the logit (sigmoid) transfer function for the output layer make sure y values are scaled from 0 to 1
# if you use the tanh for the output then you should scale between -1 and 1
# we will use sigmoid for the output layer and tanh for the hidden layer
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# derivative of sigmoid
def dsigmoid(y):
    return y * (1.0 - y)


# using tanh over logistic sigmoid is recommended
def tanh(x):
    return math.tanh(x)


# derivative for tanh sigmoid
def dtanh(y):
    return 1 - y * y


class MLP_NeuralNetwork(object):
    """
    Basic MultiLayer Perceptron (MLP) network, adapted and from the book 'Programming Collective Intelligence' (http://shop.oreilly.com/product/9780596529321.do)
    Consists of three layers: input, hidden and output. The sizes of input and output must match data
    the size of hidden is user defined when initializing the network.
    The algorithm has been generalized to be used on any dataset.
    As long as the data is in this format: [[[x1, x2, x3, ..., xn], [y1, y2, ..., yn]],
                                           [[[x1, x2, x3, ..., xn], [y1, y2, ..., yn]],
                                           ...
                                           [[[x1, x2, x3, ..., xn], [y1, y2, ..., yn]]]
    An example is provided below with the digit recognition dataset provided by sklearn
    Fully pypy compatible.
    """

    def __init__(self, input, hidden, output, iterations, learning_rate, momentum, rate_decay):
        """
        :param input: number of input neurons
        :param hidden: number of hidden neurons
        :param output: number of output neurons
        """
        # 初始化训练参数
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.rate_decay = rate_decay

        # 初始化各层层数
        self.input = input + 1  # add 1 for bias node
        self.hidden = hidden
        self.output = output

        # 设置全为1的各层向量
        self.ai = [1.0] * self.input
        self.ah = [1.0] * self.hidden
        self.ao = [1.0] * self.output

        # 创建初始化权重
        # use scheme from 'efficient backprop to initialize weights
        input_range = 1.0 / self.input ** (1 / 2)
        output_range = 1.0 / self.hidden ** (1 / 2)
        self.wi = np.random.normal(loc=0, scale=input_range, size=(self.input, self.hidden))
        self.wo = np.random.normal(loc=0, scale=output_range, size=(self.hidden, self.output))

        # 创建全为0的数组
        # 这本质上是一个临时值数组，在每次迭代时更新，基于权重在接下来的迭代中需要改变的程度

        self.ci = np.zeros((self.input, self.hidden))
        self.co = np.zeros((self.hidden, self.output))

    def feedForward(self, inputs):
        """
        The feedforward algorithm loops over all the nodes in the hidden layer and
        adds together all the outputs from the input layer * their weights
        the output of each node is the sigmoid function of the sum of all inputs
        which is then passed on to the next layer.
        :param inputs: input data
        :return: updated activation output vector
        """
        if len(inputs) != self.input - 1:
            raise ValueError('Wrong number of inputs you silly goose!')

        # input activations
        for i in range(self.input - 1):  # -1 is to avoid the bias
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.hidden):
            sum = 0.0
            for i in range(self.input):
                sum += self.ai[i] * self.wi[i][j]
            self.ah[j] = tanh(sum)

        # output activations
        for k in range(self.output):
            sum = 0.0
            for j in range(self.hidden):
                sum += self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]

    def backPropagate(self, targets):
        """
        对于输出层
        1、计算输出值与目标值之差
        2、获取sigmoid函数的导数，以确定权重需要改变的程度
        3、根据学习速率和sig导数更新每个节点的权重
        对于隐藏层
        1、计算每个输出链接的强度之和乘以目标节点必须改变的程度
        2、获取权重导数以确定需要更改的权重
        3、根据学习率和导数更改权重
        
        :param targets: y values
        :param N: learning rate
        :return: updated weights
        """
        if len(targets) != self.output:
            raise ValueError('Wrong number of targets you silly goose!')

        # 计算输出层输入的误差项
        # delta指明更改权重的方向
        output_deltas = [0.0] * self.output
        for k in range(self.output):
            error = self.ao[k] - targets[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # 计算隐藏层输入的误差项
        # delta指明更改权重的方向
        hidden_deltas = [0.0] * self.hidden
        for j in range(self.hidden):
            error = 0.0
            for k in range(self.output):
                error += output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = dtanh(self.ah[j]) * error

        # 更新连接隐藏层到输出层的权重
        for j in range(self.hidden):
            for k in range(self.output):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] -= self.learning_rate * change + self.co[j][k] * self.momentum
                self.co[j][k] = change

        # 更新连接输入层到隐藏层的权重
        for i in range(self.input):
            for j in range(self.hidden):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] -= self.learning_rate * change + self.ci[i][j] * self.momentum
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error += 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    def test(self, patterns):
        """
        Currently this will print out the targets next to the predictions.
        Not useful for actual ML, just for visual inspection.
        """
        i = 0
        for p in patterns:
            a = self.feedForward(p[0])
            # print(len(p[0]))
            # print(len(p[1]))
            a.index(max(a))
            if a.index(max(a)) == p[1].index(max(p[1])):
                i += 1
            print(p[1], '->', a.index(max(a)))

        print("准确率：", i/len(patterns))

    def train(self, patterns):
        # N: learning rate
        for i in range(self.iterations):
            error = 0.0
            random.shuffle(patterns)
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.feedForward(inputs)
                error += self.backPropagate(targets)
            with open('error.txt', 'a') as errorfile:
                errorfile.write(str(error) + '\n')
                errorfile.close()
            if i % 10 == 0:
                print('error %-.5f' % error)
            # learning rate decay
            self.learning_rate = self.learning_rate * (
                    self.learning_rate / (self.learning_rate + (self.learning_rate * self.rate_decay)))

    def predict(self, X):
        """
        return list of predictions after training algorithm
        """
        predictions = []
        for p in X:
            predictions.append(self.feedForward(p))
        return predictions


def demo():
    """
    run NN demo on the digit recognition dataset from sklearn
    """

    def load_data():
        data = np.loadtxt('data/sklearn_digits.csv', delimiter=',')

        # first ten values are the one hot encoded y (target) values
        y = data[:, 0:10]
        # y[y == 0] = -1 # if you are using a tanh transfer function make the 0 into -1
        # y[y == 1] = .90 # try values that won't saturate tanh

        data = data[:, 10:]  # x data
        # data = data - data.mean(axis = 1)
        data -= data.min()  # scale the data so values are between 0 and 1
        data /= data.max()  # scale

        out = []
        # print data.shape
        # print data.shape[0]

        # populate the tuple list with the data
        for i in range(data.shape[0]):
            fart = list((data[i, :].tolist(), y[i].tolist()))  # don't mind this variable name
            out.append(fart)

        return out

    X = load_data()
    #print(X[:len(X) - 99])
    # a = input("input a: ")
    #print(X[9])  # make sure the data looks right

    NN = MLP_NeuralNetwork(64, 20, 10, iterations=50, learning_rate=0.5, momentum=0.5, rate_decay=0.01)

    NN.train(X[:len(X) - 99])

    NN.test(X[len(X) - 99:])


if __name__ == '__main__':
    demo()
