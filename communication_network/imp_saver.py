# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 20:11:55 2017

@author: Yangyang Deng
@Email: yangydeng@163.com
"""
import tensorflow as tf
import pandas as pd
from dl_and_single_node_testing import accuracy

def continue_train(batch,step,sess,train_datas,train_labels_ohe):
    start = 0    
    for i in range(step):
        end = start+batch    
        train.run(feed_dict={x:train_datas[start:end],y_:train_labels_ohe[start:end]})
        if not i%1000:
            print('cross entory:',sess.run(cross_entropy,feed_dict={x:train_datas[start:end],y_:train_labels_ohe[start:end]}))
            #print(sess.run(learning_rate))            
        start = end
        if start+batch>=len(train_datas):
            start = 0      

if __name__=='__main__':
    train_datas = pd.read_csv('./save/train_datas.csv',header=None)
    train_labels_ohe = pd.read_csv('./save/train_labels_ohe.csv',header=None)
    test_datas = pd.read_csv('./save/test_datas.csv',header=None)
    test_labels_ohe = pd.read_csv('./save/test_labels_ohe.csv',header=None)
    
    test_datas = test_datas.values
    test_labels_ohe = test_labels_ohe.values
    train_datas = train_datas.values
    train_labels_ohe = train_labels_ohe.values
    
    
    x = tf.placeholder(tf.float32,shape=[None,4])   #保存数据特征，每条数据有4个特征，每次放入的数据可以自由设定
    y_ = tf.placeholder(tf.float32,shape=[None,10])   #保存数据标签，每次得到十条通道的OHE码。
    
    w1 = tf.Variable(tf.truncated_normal([4,30],seed=1))  
    w2 = tf.Variable(tf.truncated_normal([30,15],seed=1))
    w3 = tf.Variable(tf.truncated_normal([15,10],seed=1))
    
    hide1 = tf.nn.sigmoid(tf.matmul(x,w1))
    hide2 = tf.nn.sigmoid(tf.matmul(hide1,w2))
    hide3 = tf.nn.relu(tf.matmul(hide2,w3))
    output = tf.nn.softmax(hide3,name='output')
    
    batch =500
    sample_size = len(test_datas)
    
    global_step = tf.Variable(0)  
    learning_rate = tf.train.exponential_decay(1e-2,global_step,decay_steps=sample_size/batch,decay_rate=0.99998,staircase=True)
    cross_entropy = -tf.reduce_mean(y_*tf.log(output)) #交叉熵代价函数
    train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy,global_step=global_step)
    
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    saver.restore(sess,'./save/model.ckpt')
    
    test_result_ohe = sess.run(output,feed_dict={x:test_datas})
    accuracy(1,test_result_ohe,test_labels_ohe)
    accuracy(3,test_result_ohe,test_labels_ohe)
    
    continue_train(500,10000,sess,train_datas,train_labels_ohe)
    test_result_ohe = sess.run(output,feed_dict={x:test_datas})
    accuracy(1,test_result_ohe,test_labels_ohe)
    accuracy(3,test_result_ohe,test_labels_ohe)

    #sess.close()
    
    