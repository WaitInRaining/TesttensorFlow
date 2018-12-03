#!usr/bin/python
# coding:utf-8

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
# 导入测试数据
from tensorflow.examples.tutorials.mnist import  input_data


mnist =  input_data.read_data_sets("/temp/data", one_hot=True)
lr = 1e-3
# 在训练和测试的时候，我们想用不同的 batch_size.所以采用占位符的方式
keep_prob = tf.placeholder(tf.float32, [])
batch_size = tf.placeholder(tf.int32, [])
# 每个时刻只输入一行
input_size = 28
# 每做一次预测，需要先输入28行
timestep_size = 28
# 每个隐藏层的节点数
hidden_size = 256
# LISM layer的层数
layer_num = 5
# 最后输出分类类别数量
class_num = 10

_X = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, class_num])
# keep_prob = tf.placeholder(tf.float32)
# **步骤1：RNN 的输入shape = (batch_size, timestep_size, input_size)
X = tf.reshape(_X, [-1, 28, 28])
# # 步骤2：：定义一层 LSTM_cell，只需要说明 hidden_size, 它会自动匹配输入的 X 的维度
# lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size,forget_bias=1.0, state_is_tuple=True)
# # 步骤3：添加dropout layer ，一般只设置out_keep_prob
# lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
# #步骤4：调用MultiRNNcell实现LSTM
# mlstm_cell = rnn.MultiRNNCell([lstm_cell]*layer_num, state_is_tuple=True)
#
# # 步骤5：初始化state
# init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)


stacked_rnn = []
for iiLyr in range(layer_num):
    stacked_rnn.append(tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True))
mlstm_cell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn, state_is_tuple=True)
init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
# 步骤6：第一种
outputs = list()
state = init_state
with tf.variable_scope('RNN'):
    for timestep in range(timestep_size):
        if timestep >0 :
            # 检索当前变量名域，并将reuse设置为True s
            tf.get_variable_scope().reuse_variables()
        (cell_output, state) = mlstm_cell(X[:, timestep, :], state)
        print(cell_output)
        outputs.append(cell_output)
h_state = outputs[-1]

# 上面的输出是一个[hidden_size]的tensor,我们要分类的话，还需要一个softmax层
# 开始训练和测试
# 对w进行正态分布初始化
W= tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)
# 初始化所有偏置为0.1
bias = tf.Variable(tf.constant(0.1, shape=[class_num]), dtype=tf.float32)
# 预测出的
y_pre = tf.nn.softmax(tf.matmul(h_state, W)+bias)

#损失和评估函数 reduce_mean求均值
cross_entropy = -tf.reduce_mean(y * tf.log(y_pre))
train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
# tf.argmax返回y_pre中最大值的索引号
correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y,1))
#  tf.cast转换数据格式
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(200):
        _batch_size = 128
        batch = mnist.train.next_batch(_batch_size)
        if (i+1)%20 == 0:
            train_accuracy =sess.run(accuracy, feed_dict={_X:batch[0], y:batch[1], keep_prob:1.0, batch_size: _batch_size})
        # 已经迭代完成的 epoch 数: mnist.train.epochs_completed
            print("Iter%d, step %d, training accuracy %g" % (mnist.train.epochs_completed, (i + 1), train_accuracy))
        sess.run(train_op, feed_dict={_X: batch[0], y: batch[1], keep_prob: 0.5, batch_size: _batch_size})
    print("test accuracy %g"% sess.run(accuracy, feed_dict={
    _X: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0, batch_size:mnist.test.images.shape[0]}))


