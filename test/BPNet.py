#! usr/bin/python
# coding:utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import  matplotlib.pyplot as plt

mnist = input_data.read_data_sets("/temp/data", one_hot=True)

batch_size = 100
learning_rate = 0.01
epoches = 50

input_num_units = 28*28
output_num_units = 10
hidden_num_units = 500
# hidden_num_units = 500 一开始仅一层隐藏层为500个单元，

x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])

weights = {
    'hidden1':tf.Variable(tf.random_normal([input_num_units,hidden_num_units])),
    'hidden2': tf.Variable(tf.random_normal([hidden_num_units, hidden_num_units])),
    'hidden3': tf.Variable(tf.random_normal([hidden_num_units, hidden_num_units])),
    'hidden4': tf.Variable(tf.random_normal([hidden_num_units, hidden_num_units])),
    'hidden5': tf.Variable(tf.random_normal([hidden_num_units, hidden_num_units])),
    'output':tf.Variable(tf.random_normal([hidden_num_units, output_num_units]))
}
biases = {
    'hidden1':tf.Variable(tf.random_normal([hidden_num_units])),
    'hidden2':tf.Variable(tf.random_normal([hidden_num_units])),
    'hidden3':tf.Variable(tf.random_normal([hidden_num_units])),
    'hidden4':tf.Variable(tf.random_normal([hidden_num_units])),
    'hidden5':tf.Variable(tf.random_normal([hidden_num_units])),
    'output':tf.Variable(tf.random_normal([output_num_units]))
}
costs = list()
#第一层隐藏层
hidden_layer = tf.nn.relu(tf.add(tf.matmul(x, weights['hidden1']), biases['hidden1']))
# 第二层隐藏层
hidden_layer = tf.nn.relu(tf.add(tf.matmul(hidden_layer, weights['hidden2']), biases['hidden2']))
# 第三层隐藏层
hidden_layer = tf.nn.relu(tf.add(tf.matmul(hidden_layer, weights['hidden3']), biases['hidden3']))
# 第四层隐藏层
hidden_layer = tf.nn.relu(tf.add(tf.matmul(hidden_layer, weights['hidden4']), biases['hidden4']))
# 第五层隐藏层
hidden_layer = tf.nn.relu(tf.add(tf.matmul(hidden_layer, weights['hidden5']), biases['hidden5']))


output_layer = tf.add(tf.matmul(hidden_layer, weights['output']), biases['output'])

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

def plotCost(epoches, cost):
    epochlist = [];
    for epoch in range(epoches):
        epochlist.append(epoch+1)
    fig = plt.figure()
    plt.plot(epochlist,cost)
    plt.show()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epoches):
        avg_cost = 0
        total_batch =  int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _,c = sess.run([optimizer, cost],feed_dict={x:batch_xs,
                                                        y:batch_ys})
            avg_cost += c/total_batch

        print("Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost))
        costs.append(avg_cost)
    print("\nTraining complete!")

    pred_temp = tf.equal(tf.argmax(output_layer,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(pred_temp, tf.float32))
    print("Validation Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    plotCost(epoches, costs)



