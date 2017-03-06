# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 20:02:17 2017

@author: Administrator
"""
import tensorflow as tf
from sklearn.datasets import load_iris
import sys
sys.path.append('./')
from tools import make_OHE

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

data = load_iris()
x_data = data.data
y_data = make_OHE(data.target)


xs = tf.placeholder(tf.float32,[None,4])
ys = tf.placeholder(tf.float32,[None,3])

pre = add_layer(xs,4,3,tf.nn.softmax)   #输入层激活函数：softmax
#loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-l1),reduction_indices=[1]))
loss = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(pre),reduction_indices=[1]))  #输出层激活函数：交叉熵


train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data.reshape(150,3)})
    if(i%50==0):
        #print(sess.run(loss,feed_dict={xs:x_data,ys:y_data.reshape(150,3)}))
        print(compute_accuracy(x_data,y_data))
#sess.close()