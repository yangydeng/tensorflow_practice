# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 12:39:30 2017

@author: Administrator
"""

import tensorflow as tf

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])

product = tf.matmul(matrix1,matrix2)  #matrix mulitply np.dot(m1,m2)

##method 1 
#sess = tf.Session()
#result = sess.run(product)
#print(result)
#sess.close()


with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)