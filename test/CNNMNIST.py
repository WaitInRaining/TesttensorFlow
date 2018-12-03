#！usr/bin/python
# coding:utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
mnist = input_data.read_data_sets("/temp/data", one_hot=True)

learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# 图像的维度
n_input = 784
n_classes = 10
dropout = 0.75

x = tf.placeholder(tf.float32,[None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prop = tf.placeholder(tf.float32)


# 卷积模型
def conv2d(img, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img,w, strides=[1,1,1,1],padding='SAME'),b))
# 池化，最大化
def max_pool(img, k):
    return tf.nn.max_pool(img, ksize=[1,k,k,1],strides=[1,k,k,1],padding="SAME")

def conv_net(_X, _weight, _biases, _dropout):
    # 重新定义输入图片的模型 28*28单通道，为什么有-1？
    _X = tf.reshape(_X, shape=[-1, 28,28, 1])
    # 进行卷积，然后池化，然后dropout
    conv1 = conv2d(_X, _weight['wc1'], _biases['bc1'])
    conv1 = max_pool(conv1, k=2)
    conv1 = tf.nn.dropout(conv1, _dropout)

    # 卷积，池化，然后dropout
    conv2 = conv2d(conv1, _weight['wc2'], _biases['bc2'])
    conv2 = max_pool(conv2, k=2)
    conv2 = tf.nn.dropout(conv2, _dropout)
    # 做全连接
    dense1 = tf.reshape(conv2, [-1, _weight['wd1'].get_shape().as_list()[0]])
    dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, _weight['wd1']), _biases['bd1']))
    dense1 = tf.nn.dropout(dense1, _dropout)

    out = tf.add(tf.matmul(dense1, _weight['out']), _biases['out'])
    return out

weights = {
    'wc1':tf.Variable(tf.random_normal([5,5,1,32])), #5*5卷积，1个输入，32个输出
    'wc2':tf.Variable(tf.random_normal([5,5,32,64])),#5*5卷积，32个输入，64个输出
    'wd1':tf.Variable(tf.random_normal([7*7*64, 1024])), #全连接，7*7*64个输入，1024个输出
    'out':tf.Variable(tf.random_normal([1024, n_classes])) #1024个输入，10个输出
}

biases = {
    'bc1':tf.Variable(tf.random_normal([32])),
    'bc2':tf.Variable(tf.random_normal([64])),
    'bd1':tf.Variable(tf.random_normal([1024])),
    'out':tf.Variable(tf.random_normal([n_classes]))
}
pred = conv_net(x, weights,biases, keep_prop)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x:batch_xs,
                                       y:batch_ys,
                                       keep_prop:dropout})
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x:batch_xs,
                                                y:batch_ys,
                                                keep_prop:1.})
            loss = sess.run(cost, feed_dict={x:batch_xs,
                                             y:batch_ys,
                                             keep_prop:1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                                               y: mnist.test.labels[:256],
                                                               keep_prop: 1.}))



