#!usr/bin/python
# coding:utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/temp/data/", one_hot=True)

learning_rate = 0.01 #学习率
training_epochs = 500 #迭代次数
batch_size = 100 #批处理次数
display_step = 1 #显示步骤

# 设置图输入数据
x = tf.placeholder(tf.float32,[None, 784]) #28*28位的图片
y = tf.placeholder(tf.float32, [None, 10]) #0~9



# 设置权重和偏置
W= tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
# 激活函数
pred = tf.nn.softmax(tf.matmul(x,W) +b)

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _,c = sess.run([optimizer,cost], feed_dict={x:batch_xs,
                                                        y:batch_ys})
            avg_cost += c/ total_batch

        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
        print("optimization finished!")

        correct_prediction = tf.equal(tf.arg_max(pred, 1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
