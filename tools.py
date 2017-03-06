# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 12:46:54 2017

@author: Administrator
"""

from sklearn import preprocessing  
import tensorflow as tf

def make_OHE(names):
    data = []
    for name in names:
        data.append([name])          
    enc = preprocessing.OneHotEncoder()
    enc.fit(data)
    OHE_data = enc.transform(data).toarray()  
    return OHE_data
