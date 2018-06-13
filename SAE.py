# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 09:55:03 2018

@author: LiZhiHuan
"""
from scipy import linalg
import numpy as np
def SAE(X,S,lamb):

    A=S.dot(S.T)
    B=lamb*(X.dot(X.T))
    C=(1+lamb)*(S.dot(X.T))
    W=linalg.solve_sylvester(A,B,C)
    return W
    

    