# -*- coding: utf-8 -*-
"""
Created on Thu May 31 19:32:38 2018

@author: Administrator
"""

from sklearn import preprocessing   
import numpy as np  



def NormalizeFea(fea,mode):
    '''
    mode==0,do (X-X_mean)/X_std
    mode==1. do (X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0)) 
    '''  
    if mode==0:
        norm_fea=preprocessing.scale(fea)
    elif mode==1:
        norm_fea=np.zeros(fea.shape)
        for i in range(fea.shape[0]):
       
            max_=np.max(fea[i])
            min_=np.min(fea[i])
            norm_fea[i]=(fea[i]-min_)/(max_-min_)
    elif mode==2:
        nSmp,mFea = fea.shape
        
        feaNorm=np.sqrt(np.sum(fea*fea,1))
        
        
        b=np.zeros((nSmp,mFea))
        for ii in range(mFea):
            b[:,ii]=feaNorm
        
        
        norm_fea=fea/b
        
        
    return norm_fea

        