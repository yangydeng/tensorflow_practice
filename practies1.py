# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 10:19:26 2017

@author: Administrator
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

def add_layer(inputs,in_size,out_size,activation_fun):
    W = tf.Variable(tf.random_normal([in_size,out_size]))
    b = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    W_b = tf.matmul(inputs,W) + b
    if(activation_fun is None):
        outputs = W_b
    else:
        outputs = activation_fun(W_b)
    return outputs

def compute_accuracy(v_xs,v_ys):
    global pre
    y_pre = sess.run(pre,feed_dict={xs:v_xs})   #得到 n行 10列 矩阵，每一列表示该结果的概率。
    correct_pre = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pre,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])

pre = add_layer(xs,784,10,activation_fun=tf.nn.softmax) #输入层的激活函数为 softmax
#-------------------------
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(pre),reduction_indices=[1]))  #输出层的激活函数为交叉熵 （交叉熵可以）

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if(i%50==0):
        print(compute_accuracy(mnist.test.images,mnist.test.labels))