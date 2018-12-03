#!usr/bin/python
# coding:utf-8

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

learnRating = 1e-3
keep_prob = tf.placeholder(tf.float32, [])
batch_size = tf.placeholder(tf.int32, [])