# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 16:05:35 2018

@author: Administrator
"""

import pickle
from SAE import SAE
from NormalizeFea import NormalizeFea
import numpy as np
from math import *
from label_matrix import label_matrix
import os
from scipy import spatial
import scipy.io as scio 

a=scio.loadmat(os.getcwd()+'/'+"ImNet_2_demo_data.mat")
X_tr=a['X_tr']
X_te=a['X_te']
Y=a['Y']
#X_tr=NormalizeFea(X_tr.T,2).T

S_tr=a['S_tr']
#S_tr=NormalizeFea(S_tr,1)
S_te_pro=a['S_te_pro']
te_cl_id=a['X_te_cl_id']
Y_te=a['Y_te']

W=np.linalg.inv(np.eye(X_tr.T.dot(X_tr).shape[0])*150+X_tr.T.dot(X_tr)).dot(X_tr.T).dot(Y)
X_tr=X_tr.dot(W)
X_te=X_te.dot(W)

#fea=fea.dot(W)
#fea_tes=fea_tes.dot(W)
lamb  = 5;

W=SAE(X_tr.T,S_tr.T,lamb).T


##计算余弦距离并标准化 
S_te_est = X_te .dot(W)
#S_te_est = X_te .dot(NormalizeFea(W,2))
dist     =  1 - spatial.distance.cdist(S_te_est,S_te_pro,'cosine')
dist     = NormalizeFea(dist,0)
#[F --> S], projecting data from feature space to semantic space 
HITK=5
Y_hit5 =np. zeros((dist.shape[0],HITK))
for i in range(dist.shape[0]):
    I=np.argsort(dist[i])[::-1]
    Y_hit5[i,:]=np.squeeze(te_cl_id[I[0:HITK]])


n=0
for i in range(dist.shape[0]):
    if Y_te[i] in Y_hit5[i,:]:
        n=n+1

zsl_accuracy = n/dist.shape[0]
print(zsl_accuracy)   
#[S --> F], projecting from semantic to visual space 
#dist    =  1 - zscore(pdist2(X_te, (S_te_pro * W'), 'cosine')) ;
dist     =  1 - spatial.distance.cdist(X_te,S_te_pro.dot(W.T),'cosine')
dist     = NormalizeFea(dist,0)
HITK=5
Y_hit5 =np. zeros((dist.shape[0],HITK))
for i in range(dist.shape[0]):
    I=np.argsort(dist[i])[::-1]
    Y_hit5[i,:]=np.squeeze(te_cl_id[I[0:HITK]])



n=0
for i in range(dist.shape[0]):
    if Y_te[i] in Y_hit5[i,:]:
        n=n+1

zsl_accuracy = n/dist.shape[0]
print(zsl_accuracy)  



     


