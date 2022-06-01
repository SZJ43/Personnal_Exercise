# --coding:utf-8 --
import datetime
import neuralNet
import numpy as np

# 加载训练数据
training_data_file = open("data/mnist_train2.csv", "r")
training_data_list = training_data_file.readlines()
training_data_file.close()

# 参数初始化
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
n = neuralNet.neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 开始训练
time_begin = datetime.datetime.now()
print(str(time_begin))

for record in training_data_list:
    # 以逗号分割每条记录
    all_values = record.split(',')
    # 对输入进行处理
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # 创建目标输出值
    targets = np.zeros(output_nodes) + 0.01
    # 第一项是此条记录的目标标签
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)
    pass

time_end = datetime.datetime.now()
print(str(time_end))
time = time_end - time_begin
print('Time Elapsed: ' + str(time))
print("Training Completed")


# 加载测试数据
test_data_file = open("data/mnist_test2.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# 记录神经网络的表现
scorecard = []

# 开始测试
for record in test_data_list:
    # 以逗号分割每条记录
    all_values = record.split(',')
    # 第一个标签是正确值
    correct_label = int(all_values[0])
    # 预处理输入数据
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # 检验神经网络
    outputs = n.query(inputs)
    # 取得最高值的索引对应着其标签
    label = np.argmax(outputs)
    # 将结果分类为正确与不正确，添加入记录列表
    if label == correct_label:
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

# 计算平均表现分数
scorecard_array = np.asarray(scorecard)
print('Performance = ', scorecard_array.sum() / scorecard_array.size)
