# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 21:03:25 2017

@author: Administrator
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import tempfile

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import tensorflow as tf


mnist = read_data_sets("MNIST_data/",one_hot=True) 
     
#  MNIST 入门版
##minist.train.images.shape => (55000,784)  表示有55000张图,每张图有784个像素点

#x = tf.placeholder(tf.float32,[None,784])
#
#W = tf.Variable(tf.zeros([784,10]))
#b = tf.Variable(tf.zeros([10]))
#
#y = tf.nn.softmax(tf.matmul(x,W) + b)       #y:预测的结果
#
#y_ = tf.placeholder(tf.float32,[None,10])       #y_:实际的结果
#cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#
#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#
#init = tf.global_variables_initializer()
#
#sess = tf.Session()
#sess.run(init)
#
#correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#
#for i in range(1000):
#    batch_xs,batch_ys = mnist.train.next_batch(100)
#    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
#    print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))


#------------------------------------------------------------------------------------------------------------


#   MNIST     深入版
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)
cross_entropy = -tf.reduce_sum(y_*tf.log(y))


sess.run(tf.global_variables_initializer())

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

correct_predition = tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_predition,tf.float32))    

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(50)
    train_step.run(feed_dict={x:batch_xs,y_:batch_ys})
    print(accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels}))
    
    
    
    

















