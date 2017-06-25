# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 13:17:52 2017

@author: Yangyang Deng
@Email: yangydeng@163.com
"""

import tensorflow as tf
import numpy as np

def generate_real_label(data,has_one_open):
    if has_one_open:
        real_label.append([0])
        return True
    
    day_time = data[0]
    tunnel = data[1]
    is_weekend = data[2]
    weather = data[3]
    #feeling = data[4]
    
    if day_time==tunnel:
        
        if tunnel>=8:
            real_label.append([1])            
            
        elif weather==0:
            real_label.append([1])            
            
        elif (tunnel==6 or tunnel==7) and is_weekend==1:
            real_label.append([1])        
            
        else:
            real_label.append([1] if rds.randint(0,10)>2 else [0])        
        return True if real_label[-1]==[1] else False
    else:
        real_label.append([0] if rds.randint(0,10)>0 else [1])
        return True if real_label[-1]==[1] else False
    
def generate_ohe(labels):
    res = []
    while labels:
        max_index = labels[:10]
        ohe_index = [i[0] for i in max_index]
        for i in range(10):
            res.append(ohe_index)
        labels = labels[10:]    
    return res        

def accuracy(rank,test_result_ohe,test_labels_ohe):
    count = 0
    for i in range(len(test_result_ohe)):
        dic = {}
        for key,value in enumerate(test_result_ohe[i]):
            dic[key] = value
        L = sorted(dic.items(),key=lambda item:item[1]) #sorted之后返回的是list
        L.reverse()
        
        for j in range(rank):
            if np.argmax(test_labels_ohe[i]) == L[j][0]:
                count+=1        
    print(str(rank)+'次命中准确率为：',count/(i+1))        


if __name__=='__main__':
    rds = np.random.RandomState(1)
    real_label=[]
    datas = []          
    for day in range(1000):
        for day_time in range(10):
            has_one_open = False
            for tunnel in range(10):
                #datas：daytime：当天的时间段；is_weekend:1是，0否；天气；0晴，1阴，2雨；心情：1好，0坏。
                data = [day_time,tunnel,0 if (day%6 or day%7) else 1,rds.randint(0,3)]
                #data.append(rds.randint(0,2)) #心情，可被注释
                datas.append(data) 
                if tunnel==9 and not has_one_open: #如果前8个信道都不同，则第9个信道为通。
                    real_label.append([1])
                    continue
                else:
                    has_one_open = generate_real_label(data,has_one_open)
    
    train_datas = datas[:int(0.7*len(datas))]
    test_datas = datas[int(0.7*len(datas)):]
    
    train_labels = real_label[:int(0.7*len(datas))]
    test_labels = real_label[int(0.7*len(datas)):]
    
    
    
    train_labels_ohe = generate_ohe(train_labels)
    test_labels_ohe = generate_ohe(test_labels)
    
                
    
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
    
    start = 0
    step = 10000
    
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    #saver = tf.train.Saver() 
    
    for i in range(step):
        end = start+batch    
        train.run(feed_dict={x:train_datas[start:end],y_:train_labels_ohe[start:end]})
        if not i%1000:
            print('cross entory:',sess.run(cross_entropy,feed_dict={x:train_datas[start:end],y_:train_labels_ohe[start:end]}))
            #print(sess.run(learning_rate))            
        start = end
        if start+batch>=len(train_datas):
            start = 0        
    
    test_result_ohe = sess.run(output,feed_dict={x:test_datas})
    
    accuracy(1,test_result_ohe,test_labels_ohe) #一次成功率
    accuracy(3,test_result_ohe,test_labels_ohe) #三次成功率
    #saver.save(sess,'./save/model.ckpt')
    sess.close()

