# -*- coding: utf-8 -*-
"""
Created on Mon May 22 12:28:13 2017

@author: Yangyang Deng
@Email: yangydeng@163.com

本文件为 《Tensorflow官方文档 v-1.2》中，构建一个多层神经网络案例，附注释。
"""

import tensorflow as tf 
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

mnist = read_data_sets("MNIST_data/",one_hot=True)
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32,shape=[None,784])   # 保存图片
y_ = tf.placeholder(tf.float32,shape=[None,10])   # 保存图片真实的 label


def weight_variable(shape): #weight initial
    initial = tf.truncated_normal(shape,stddev=0.1,seed=1)  
    return tf.Variable(initial)

def bias_variable(shape):  #bias initial
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):  # convolution
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")   #strides:移动调节，1.batch 2.height 3.width 4.channels
    #strides=[1,1,1,1]: 每次卷积一个批次，高和宽的移动为一步，每次卷积一个频道（频道为RGB三色道，若为彩色：3，若为黑白：1）


def max_pool_2x2(x): # pooling
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')   #ksize:池化窗口设置 同上 strides
    # ksize=[1,2,2,1] 每次池化一个批次，池化窗口2*2，每次池化一个频道
    # 由于池化窗口的设置为2*2，所以池化窗口的移动也是 2,2


# 第一层 卷积+池化
W_conv1 = weight_variable([5,5,1,32])  # 5*5的局部感受野，1个输入频道，32个输出频道（32个卷积核）
b_conv1 = bias_variable([32])          
x_image = tf.reshape(x,[-1,28,28,1])    # -1：为了把原来的向量铺平，28*28：图片的长宽，1:颜色通道 （RBG：3，黑白：1）

h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)    # x*W+b
h_pool1 = max_pool_2x2(h_conv1)  # max pooling

# 第二层 卷积+池化
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 第三层 全连层，输出为1024
W_fcl = weight_variable([7*7*64,1024])  #此时，照片的尺寸减小到7*7，（经过两次max_pooling,28*28 的长和宽经过两次除以2，所以变为7*7）
b_fcl = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64]) #将第二个卷基层铺平
h_fcl = tf.nn.relu(tf.matmul(h_pool2_flat,W_fcl)+b_fcl)

# dropout
keep_prob = tf.placeholder(tf.float32) # keep_prob: 一个神经元的输出在dropout中保持不变的概率。
h_fcl_drop = tf.nn.dropout(h_fcl,keep_prob) # dropout 

# 第四层，输出层softmax，输出层也是全连层，但是固定输出端口只能为10个，便于与真实结果匹配。
W_fc2 = weight_variable([1024,10]) 
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fcl_drop,W_fc2)+b_fc2) # 套上softmax

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv)) #交叉熵代价函数
train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_pre = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_pre,tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(200):
    batch = mnist.train.next_batch(50)
    if i%100==0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
        #print(batch[1][0])
        print("step %d, training accuracy %g"%(i,train_accuracy))
    train.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5}) #在训练时，dropout才生效
    
print ("test accuracy %g"%accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0})) #在实际工作中，dropout不生效。

sess.close()


