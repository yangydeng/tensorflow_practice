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
import pandas as pd

train_x = pd.read_csv('./csv/train_x_03_01_3.csv').values
train_y = pd.read_csv('./csv/train_y_03_01_3.csv').ix[:,0].values


def add_layer(inputs,in_size,out_size,activation_fun):
    W = tf.Variable(tf.random_normal([in_size,out_size],seed=1,stddev=0.01))
    b = tf.Variable(tf.zeros([1,out_size]) + 0.01)
    W_b = tf.matmul(inputs,W) + b
    if(activation_fun is None):
        outputs = W_b
    else:
        outputs = activation_fun(W_b)
    return outputs

def compute_accuracy(v_xs,v_ys):
    global pre
    y_pre = sess.run(pre,feed_dict={xs:v_xs})   #得到 n行 10列 矩阵，每一列表示该结果的概率。
    correct_pre = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))    #correct_pre 是true or false 的结果，tf.argmax()得到最大值对应的下标
    accuracy = tf.reduce_mean(tf.cast(correct_pre,tf.float32))      #此处必须为tf.float32的格式，若为tf.int32形式，则mean()之后的结果为0
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

#data = load_iris()
#x_data = data.data
#y_data = make_OHE(data.target)
x_data = train_x
y_data = train_y

xs = tf.placeholder(tf.float32,[None,4])
ys = tf.placeholder(tf.float32,[None,3])

#l1 = add_layer(xs,4,2,None)
pre = add_layer(xs,4,3,tf.nn.softmax)   #输入层为特征：xs, 输出层为激活函数为softmax，用于前向传播。
#loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-l1),reduction_indices=[1]))
loss = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(pre),reduction_indices=[1]))  #交叉熵作为代价函数，用于反向传播。


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
