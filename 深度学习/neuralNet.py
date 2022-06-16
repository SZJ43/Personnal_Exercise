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
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.inodes, self.hnodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.hnodes, self.onodes))

        # 用sigmoid作为激活函数
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self, inputs_list, targets_list):
        # 将输入和输出转换成二维数组
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # 计算’输入层向量与输入层到隐藏层连接的权重向量‘的点积
        hidden_inputs = np.dot(self.wih, inputs)
        # 计算‘隐藏层的激活’
        hidden_outputs = self.activation_function(hidden_inputs)

        # 计算‘隐藏层(输出)向量与隐藏层到输出层连接的权重向量’的点积
        final_inputs = np.dot(self.who, hidden_outputs)
        # 计算‘输出层的激活’
        final_outputs = self.activation_function(final_inputs)

        # 均方误差对每个输出的偏导：输出层各个输出的误差组成的向量
        outputlayers_out_errors = final_outputs - targets

        # 公式1：输出层各个输入的误差组成的向量
        outputlayers_in_errors = np.dot(outputlayers_out_errors, final_outputs * (1.0 - final_outputs))

        # 公式2：通过反向传播更改由隐藏层到输出之间的权重矩阵
        self.who -= self.lr * np.dot(outputlayers_in_errors, np.transpose(hidden_outputs))

        # 公式3：隐藏层各个输出的误差组成的向量
        hiddenlayers_out_errors = np.dot(self.who.T, outputlayers_in_errors)

        # 公式4：隐藏层各个输入的误差组成的向量
        hiddenlayers_in_errors = np.dot(1 - hidden_outputs * hidden_outputs, hiddenlayers_out_errors)
        
        # 公式5：通过反向传播更改由输入层到隐藏层之间的权重矩阵
        self.wih -= self.lr * np.dot(hiddenlayers_in_errors, np.transpose(inputs))
        pass

    # 用更新后的参数输出结果
    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
