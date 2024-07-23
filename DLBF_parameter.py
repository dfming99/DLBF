# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 10:28:06 2022

@author: admin
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

import sys 
import sympy
import math

#from sympy import *
from math import e
from math import floor
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier

#feature_matrix = pd.read_csv('./attack_URL_data_features_all.csv',encoding = 'ISO-8859-1')
#feature_matrix = pd.read_csv('./dataset_3.csv',encoding = 'ISO-8859-1')
feature_matrix = pd.read_csv('./dataset_4.csv',encoding = 'ISO-8859-1')

len_train_good = len(feature_matrix[feature_matrix['label']==0][:int(0.4*len(feature_matrix[feature_matrix['label']==0]))])  
len_train_mali = len(feature_matrix[feature_matrix['label']==1]) 

training=pd.concat([feature_matrix[feature_matrix['label']==0][:len_train_good],feature_matrix[feature_matrix['label']==1]])
training = training.reset_index(drop=True)
training_nourl = training.drop(columns=['url'])
x_train = training_nourl.iloc[:,:-1]
x_label = training_nourl.iloc[:,-1]

t = np.arange(0,1,0.02)
x = sympy.symbols('x')  #x means lambda
M_star = 400000
region = 5
H_url = []
K = 200
beta = math.log(e,2)
log2 = math.log(2)
for i in range(K):
    H_url.append(training.loc[i,'url'])


def compute_average_size(dataset):
    sum = 0
    for i in range(len(dataset)):
        sum += len(dataset.loc[i,'url'])
        
    return sum/len(dataset)

def Random_forest(x_train,x_label):
    print(" RF_classifier training...")
    clf = RandomForestClassifier(n_estimators=128,max_leaf_nodes=20, random_state=0)  #max_leaf_nodes
    clf.fit(x_train, x_label)
    train_answer = clf.predict_proba(x_train)
    #test_answer = clf.predict_proba(test_v)
    #fp_test = clf.predict_proba(test_v)[:,0]
    #print("test_answer = ",clf.predict_proba(test_v))
    return train_answer, clf
'''def f(x):
    
    f = -1*M_star  #initial value
    alpha = compute_average_size(training.loc[:len_train_good,'url'])
    for i in range(len(region)):
        replace = log(2)*x*beta*len_train_mali*B_distribution[i]/(Q_distribution[i]+x*alpha*len_train_good*H_distribution)
        f += beta*len_train_mali*B_distribution[i]*(((log(2)*x*alpha*K*H_distribution[i])/Q_distribution[i]+x*alpha*len_train_good*H_distribution[i])-(log(replace,2)))
    return f'''

def Newton_RM(M_star,B_distribution,Q_distribution,H_distribution):
    x0 = 0.1
    counter = 0
    x_list = [x0]
    f = -1*M_star  #initial value
    alpha = compute_average_size(training.loc[:len_train_good,:])
    for i in range(region):
        replace = log2*x*beta*len_train_mali*B_distribution[i]/(Q_distribution[i]+x*alpha*len_train_good*H_distribution[i])
        #print(replace)
        f += beta*len_train_mali*B_distribution[i]*(((log2*x*alpha*K*H_distribution[i])/Q_distribution[i]+x*alpha*len_train_good*H_distribution[i])-(sympy.log(replace,2)))
    #print(f.evalf(subs={x:1}))
    #print(sympy.diff(f,x).evalf(subs={x:1}))
    #print(sympy.nsolve(f,2))
    #print(sympy.diff(f,x))
    while True:
        if sympy.diff(f,x).evalf(subs = {x:x0}) == 0:
            #print('extreme point: {}'.format(x0))
            break
        else:
            x0 = x0-f.evalf(subs={x:x0})/sympy.diff(f,x).evalf(subs={x:x0})
            x0_i = complex(x0)
            x0 = math.sqrt(x0_i.real**2+x0_i.imag**2)
            x_list.append(x0)
            
            #print(type(x0))
            #print("convert to complex:",complex(x0))
        if len(x_list)>1:
            counter += 1
            #print(counter)
            #print(x_list[-1])
            error = abs((x_list[-1]-x_list[-2])/x_list[-1])
            #print(error)
            if error < 10**(-2):
                #print('end!')
                break
        else: 
            pass
        
    return x_list[-1]

def flush_distribution(B,Q,H):
    for i in range(len(B)):
        B[i] = 0
        Q[i] = 0
        H[i] = 0
        
    return B,Q,H

if __name__ == '__main__':
    
    B_distribution = []  #malicious URL distribution 
    Q_distribution = []  #benign URL distribution 
    H_distribution = []
    false_p = 0
    min_false_p = 1000000
    thresholds = []
    
    for i in range(region):
        B_distribution.append(0)
        Q_distribution.append(0)
        H_distribution.append(0)
    #average size of benign url 
    all_combinations_threshold = list(combinations(range(1,len(t)+1),region-1))
    
    for i in range(len(all_combinations_threshold)):
        all_combinations_threshold[i] = list(all_combinations_threshold[i])
        for j in range(len(all_combinations_threshold[i])):
            #print(all_combinations_threshold[i][j])
            all_combinations_threshold[i][j] = all_combinations_threshold[i][j]/len(t)
    #print(len(all_combinations_threshold))
    train_answer,rf_classifier = Random_forest(x_train, x_label)
    train_answer = train_answer[:,-1]
    for i in range(0,floor(0.05*len(all_combinations_threshold))):
        B_distribution, Q_distribution, H_distribution = flush_distribution(B_distribution, Q_distribution, H_distribution)
        for j in range(len_train_good):
            for k in range(1,len(all_combinations_threshold[i])):           #benign URL density
                if train_answer[j] >= 0 and train_answer[j] < all_combinations_threshold[i][0]:
                    Q_distribution[0] += 1/len_train_good
                    break
                elif train_answer[j] >= all_combinations_threshold[i][-1] and train_answer[j] <= 1:
                    Q_distribution[-1] += 1/len_train_good
                    break
                elif train_answer[j] >= all_combinations_threshold[i][k-1] and train_answer[j] < all_combinations_threshold[i][k]:
                    Q_distribution[k] += 1/len_train_good
                    break
        for j in range(len_train_good,len_train_good+len_train_mali):       #malicious URL density
            for k in range(1,len(all_combinations_threshold[i])):
                if train_answer[j] >= 0 and train_answer[j] < all_combinations_threshold[i][0]:
                    B_distribution[0] += 1/len_train_mali
                    break
                elif train_answer[j] >= all_combinations_threshold[i][-1] and train_answer[j] <= 1:
                    B_distribution[-1] += 1/len_train_mali
                    break
                elif train_answer[j] >= all_combinations_threshold[i][k-1] and train_answer[j] < all_combinations_threshold[i][k]:
                    B_distribution[k] += 1/len_train_mali
                    break
        for j in range(K):                                                  #frequently toured URL density
            for k in range(1,len(all_combinations_threshold[i])):           #benign URL density
                if train_answer[j] >= 0 and train_answer[j] < all_combinations_threshold[i][0]:
                    H_distribution[0] += 1/K
                    break
                elif train_answer[j] >= all_combinations_threshold[i][-1] and train_answer[j] <= 1:
                    H_distribution[-1] += 1/K
                    break
                elif train_answer[j] >= all_combinations_threshold[i][k-1] and train_answer[j] < all_combinations_threshold[i][k]:
                    H_distribution[k] += 1/K
                    break
        lambda_i = Newton_RM(M_star,Q_distribution,B_distribution,H_distribution)
        lambda_i = complex(lambda_i)
        #print(complex(lambda_i))
        lambda_i = math.sqrt(lambda_i.real*lambda_i.real + lambda_i.imag*lambda_i.imag)
        alpha = compute_average_size(training.loc[:len_train_good,:])
        false_p = 0
        epsilon_i = [0,0,0,0,0]
        for l in range(region):
            
            if Q_distribution[l]+lambda_i*alpha*len_train_good*H_distribution[l] != 0:
                
                epsilon_i[l] = log2*lambda_i*math.log(math.e,2)*B_distribution[l]*len_train_mali/(Q_distribution[l]+lambda_i*alpha*len_train_good*H_distribution[l])
                #if i == 40:
                    #print('epsilon i is',epsilon_i)
                false_p += Q_distribution[l]*epsilon_i[l]
            else:
                pass
        if false_p <= min_false_p:
            min_false_p = false_p
            thresholds = all_combinations_threshold[i]
            min_epsilon_i = epsilon_i
            
            #print(min_false_p)
        #print(i)
        #print('i is {}'.format(i))
        #print(thresholds)
    output_path = './result_4.txt'
    
    with open(output_path,'a',encoding = 'utf-8') as file1:
        print('min_false_p is {}, thresholds are {}, epsilon_i is {}'.format(min_false_p,thresholds,min_epsilon_i), file = file1)
    #print('optimal thresholds are:',thresholds)
        
    #train_answer,rf_classifier = Random_forest(x_train, x_label)
    
    