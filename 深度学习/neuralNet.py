# --coding:utf-8 --
import numpy as np
import scipy.special


# 定义神经网络的结构
class neuralNetwork:
    # 初始化神经网络
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningrate):
        # 设置输入层，隐藏层、输出层的节点数
        self.inodes = inputNodes
        self.hnodes = hiddenNodes
        self.onodes = outputNodes

        # 学习率
        self.lr = learningrate

        # 设置输入层和输出层权重矩阵
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # 用sigmoid作为激活函数
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self, inputs_list, targets_list):
        # 将输入和输出转换成二维数组
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # 计算隐藏层的输入
        hidden_inputs = np.dot(self.wih, inputs)
        # 计算隐藏层的输出信号
        hidden_outputs = self.activation_function(hidden_inputs)

        # 计算输出层的输入
        final_inputs = np.dot(self.who, hidden_outputs)
        # 计算输出层的最终输出
        final_outputs = self.activation_function(final_inputs)

        # 输出层误差
        output_errors = targets - final_outputs
        # 隐藏层误差
        hidden_errors = np.dot(self.who.T, output_errors)

        # 反向传播更新参数
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        pass

    # 用更新后的参数输出结果
    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
