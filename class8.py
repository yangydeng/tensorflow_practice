# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 13:16:36 2017

@author: Administrator
"""

import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

for i in range(10):
    output = tf.multiply(input1,input2)
    with tf.Session() as sess:
        print(sess.run(output,feed_dict={input1:[i],input2:[2.0]})) # Variable 与 placeholder 的不同？