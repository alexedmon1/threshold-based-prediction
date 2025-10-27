# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 16:02:16 2018

@author: edmondsd

Use finaldf from SVM_for_R1.py
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def CompareX(x0, x1):
    result = 0
    if x0 > x1:
        result = 0
    if x1 > x0:
        result = 1    
    return result

def plot_errors(thresh, d):
    """
    Dictionary should have 2 values for each statistic. 
    First value is ERROR0
    Second value is ERROR1
    """
    N=len(d.keys())
    ind = np.arange(N)    # the x locations for the groups
    width = 0.35 
    ERROR1 = []
    ERROR0 = []
    for key in d.keys():
        ERROR1.append(d[key][1])
        ERROR0.append(d[key][0])
      # the width of the bars: can also be len(x) sequence
    p1 = plt.bar(ind, ERROR1, width, yerr = 0)
    p2 = plt.bar(ind, ERROR0, width, bottom=ERROR1, yerr = 0)

    plt.ylabel('Errors')
    plt.title(thresh)
    plt.xticks(ind, ('median', 'tvar', 'skew', 'tmin', 'tmax'))
    plt.yticks(np.arange(0, 81, 10))
    plt.legend((p1[0], p2[0]), ('Error 1', 'Error 0'))
    plt.savefig('D:/F31/Results/SVM_Freesurfer/error_analysis/errorplot'+str(TARGET)+str(thresh)+'_fs.png')
    plt.close() 

def plot_bagging_dist(d):
    """
    Dictionary should have 2 values for each statistic. 
    First value is ERROR0
    Second value is ERROR1
    """
    N=len(d.keys())
    ind = np.arange(N)    # the x locations for the groups
    width = 0.35 
    ERROR1 = []
    ERROR0 = []
    KEYS = []
    for key in d.keys():
        ERROR1.append(d[key][1])
        ERROR0.append(d[key][0])
        KEYS.append(key)
      # the width of the bars: can also be len(x) sequence
    p1 = plt.bar(ind, ERROR1, width, yerr = 0)
    p2 = plt.bar(ind, ERROR0, width, bottom=ERROR1, yerr = 0)

    plt.ylabel('Errors')
    plt.title('Bagging errors')
    plt.xticks(ind, (KEYS))
    plt.yticks(np.arange(0, 81, 10))
    plt.legend((p1[0], p2[0]), ('Error 1', 'Error 0'))
    plt.plot()    
    plt.savefig('D:/F31/Results/SVM_Freesurfer/error_analysis/distplot_bagging'+str(TARGET)+'_fs.png')
    plt.close() 

def CreateObs(ERROR0, ERROR1):
    comb = (ERROR0 + ERROR1)/2
    obs = np.array([[ERROR0, comb],[ERROR1, comb]])
    return obs

def diffprop(obs):
    """
    `obs` must be a 2x2 numpy array.

    Returns:
    delta
        The difference in proportions
    ci
        The Wald 95% confidence interval for delta
    corrected_ci
        Yates continuity correction for the 95% confidence interval of delta.
    """
    n1, n2 = obs.sum(axis=0)
    prop1 = obs[0,0] / n1
    prop2 = obs[1,0] / n2
    delta = prop1 - prop2

    # Wald 95% confidence interval for delta
    se = np.sqrt(prop1*(1 - prop1)/n1 + prop2*(1 - prop2)/n2)
    ci = (0 - 1.96*se, 0 + 1.96*se)

    # Yates continuity correction for confidence interval of delta
    correction = 0.5*(1/n1 + 1/n2)
    corrected_ci = (ci[0] - correction, ci[1] + correction)

    return delta, ci, corrected_ci

def convert(val):
    val2 = val.strip('[]')
    val3 = list(val2.split(','))
    val4 = [int(val3[0]), int(val3[1])]
    return val4


def plot_bagging_errors(df, ci1_list, ci0_list, delta_list):
    plt.fill_between(np.array(df.index.unique()), np.array(ci1_list), np.array(ci0_list), alpha = 0.1, color = 'grey', label = '95% Confidence Interval')
    plt.scatter(df.index.unique(), delta_list, c = 'red', label = 'Proportion between Obs and Exp')
#    plt.xlim([0, df.index.unique()[-1]+0.01])
#    plt.plot(ci1_list, linestyle = 'dashed', c = 'black')
#    plt.plot(ci0_list, linestyle = 'dashed', c = 'black')
    plt.ylim([-1, 1.1])
#    plt.xlim([-1,df.index.unique()[-1]+1])
    plt.xlabel('Threshold')
    plt.ylabel('Difference in Proportion')       
    plt.title('Deviation from balanced prediction error')
    plt.legend(loc = 2)
    plt.show()
    plt.savefig('D:/F31/Results/SVM_Freesurfer/error_analysis/errorplot_bagging'+str(TARGET)+'_fs.png')
    plt.close()
    
#%%
statistics = ['median', 'tvar', 'skew', 'tmin', 'tmax']
d = {}   
delta_list = []
ci0_list = []
ci1_list = [] 
for thresh in finaldf.index.unique():
    df = finaldf.loc[thresh]
    ERROR0 = 0
    ERROR1 = 0
    CORRECT = 0
    for index, row in df.iterrows(): 
        x0 = 0
        x1 = 0
        result = 0
        for val in row[0:len(df.columns)]:
            if val == 0:
                x0 += 1
            if val == 1:
                x1 += 1                   
        result = CompareX(x0, x1)
        vector = pd.Series([index, row[-1], result])
        if vector[1] == vector[2]:
            CORRECT += 1
        elif vector[1] > vector[2]: 
            ERROR1 += 1 # Predicted 1 when actually 0 (Type 1 error)
        elif vector[1] < vector[2]:
            ERROR0 += 1 # Predicted 0 when actually 1 (Type 2 error)
    d[thresh] = [ERROR0, ERROR1]
    obs = CreateObs(ERROR0, ERROR1) 
    print(obs)
    delta, ci, corrected_ci = diffprop(obs)
    delta_list.append(delta)
    ci0_list.append(corrected_ci[0])
    ci1_list.append(corrected_ci[1])
    print(str(thresh) +" bagging")
    print('CORRECT: ' + str(CORRECT))
    print('ERROR0: ' + str(ERROR0))
    print('ERROR1: ' + str(ERROR1))
plot_bagging_dist(d)
plot_bagging_errors(finaldf, ci1_list, ci0_list, delta_list)

#%%
obs = np.array([[44, 45], [44, 45]])
for val in df2.index:
    obs = np.array([[44, 45],[44, 45]])
    delta, ci, corrected_ci = diffprop(obs)
    ci0_list.append(ci[0])
    ci1_list.append(ci[1])    

for val in df2.index:
    obs = np.array([convert(df2.loc[val][0]),[44, 45]])
    delta, ci, corrected_ci = diffprop(obs)
    delta_list.append(delta)

    
