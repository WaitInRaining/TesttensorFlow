#!usr/bin/python
# coding:utf-8

import  numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import matplotlib.pyplot as plt

# 超参数定义
numstep = 5  #只能记忆五步
batch_size = 200
num_class = 2;
state_size = 4
learning_rate = 0.1

# 生成数据
def gen_data(size=1000000):
    X = np.ndarray(np.random.choice(2, size=(size,1)))
    Y= []
    for i in range(size):
        threshold = 0.5
        if X[i-3] == 1:
            threshold +=0.5
        if X[i-8] == 1:
            threshold -= 0.25
        if np.random.rand() > threshold:
            Y.append(0)
        else:
            Y.append(1)
    return X, np.ndarray(Y)

#  生成批次数据: 原始数据，片段程度，步数
def gen_batch(raw_data, batch_size, num_step):
    raw_x, raw_y = raw_data
    data_length = len(raw_x)
    batch_pation_length = data_length // batch_size #分成的片段数 5000
    data_x = np.zeros([batch_size, batch_pation_length], dtype=np.int32)# 200 * 5000
    data_y = np.zeros([batch_size, batch_pation_length], dtype= np.int32)# 200 * 50000
    for i in range(batch_size):
        data_x = raw_x[batch_pation_length*i: batch_pation_length*(i+1)]
        data_y = raw_y[batch_pation_length*i: batch_pation_length*(i+1)]
    epoch_size = batch_pation_length // num_step #迭代次数
    for i in range(epoch_size):
        x = data_x[:, i*num_step:(i+1)*num_step]
        y = data_y[:, i*num_step:(i+1)*num_step]
        yield (x,y) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def gen_epochs(n, num_steps):
    for i in range(n):
        yield gen_batch(gen_data(),batch_size, num_step=num_steps)


#定义输入输出格式
x = tf.placeholder(tf.int32, [batch_size, numstep], name='x') # 200*5
y = tf.placeholder(tf.int32, [batch_size, numstep], name='y') # 200*5
init_state = tf.zeros([batch_size, state_size])  #初始化状态， 200*4
# rnn输入
x_one_hot = tf.one_hot(x, num_class)
rnn_input = tf.unstack(x_one_hot, axis=1)
#RNN输入
with tf.Variable_scope('rnn_cell'):
    W = tf.get_variable('W', [num_class + state_size, state_size])
    b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))
    



