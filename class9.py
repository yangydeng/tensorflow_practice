# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 13:22:36 2017

@author: Administrator
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+ 0.1)
    Wx_plus_b = tf.matmul(inputs,Weights) + biases
    if(activation_function is None):
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data)- 0.5 + noise

xs = tf.placeholder(tf.float32,[None,1])    #无论给出多少个sample都可以
ys = tf.placeholder(tf.float32,[None,1])    #placeholder 相当于定义了变量但没有初始化

l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)  #输入层激活函数为：relu
predition = add_layer(l1,10,1,activation_function=None) #隐层激活函数为：线性函数

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predition),reduction_indices=[1]))   #输出层激活函数为：交叉熵函数

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()


for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if(i%50==0):
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass 
        #print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        predition_value = sess.run(predition,feed_dict={xs:x_data})
        lines = ax.plot(x_data,predition_value,'r-',lw=5)
        print('loss: ',sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        plt.pause(0.1)
        
        
        