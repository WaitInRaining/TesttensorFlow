#!usr/bin/python
# coding:utf-8

import tensorflow as tf

# hello_op = tf.constant('hello tensorflow!') #定义一个常量
# a = tf.constant(10)
# b = tf.constant(32)
# compute_op = tf.add(a,b)
#
# #启动TensorFlow的session，调用session的run方法启动整个graph
# with tf.Session() as sess:
#     print(sess.run(hello_op))
#     print(sess.run(compute_op))
#     print(sess.run(a ** b))

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
add = tf.add(a, b)
mul = tf.multiply(a,b)

matrix1 = tf.constant([[3.,3.]])
matrix2 = tf.constant([[2.], [2.]])
product = tf.matmul(matrix1, matrix2)
c = tf.Variable(tf.random_normal([5, 5, 1, 32]))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    print(c)
    print(sess.run(add, feed_dict={a:2, b:3}))
    print(sess.run(mul, feed_dict={a:2, b:3}))
    print(sess.run(product))

