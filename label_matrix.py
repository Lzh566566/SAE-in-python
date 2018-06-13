# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 10:20:09 2018

@author: Administrator
"""
import numpy as np
def label_matrix(label_vector):
    n=label_vector.shape[0]#样本数
    Y=label_vector
    labels=np.unique(Y)
    c=len(labels)#类别数
    label_mat=np.zeros((n,c))
    cls=0
    for i in range(n):
        if label_vector[i]==labels[cls]:
            label_mat[i][cls]=1
        else:
            cls=cls+1
            label_mat[i][cls]=1
    return label_mat


            
        