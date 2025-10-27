# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 19:41:31 2018

@author: edmondsd
"""
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('D:/F31/Results/SVM_Freesurfer/27_updated/age_27_ns.csv')
df.columns = ['threshold', 'median', 'tvar', 'skew', 'ten', 'ninety']
df2 = df[['threshold', 'median']]
df2.set_index('threshold', inplace = True)

def convert(val):
    val2 = val.strip('[]')
    val3 = list(val2.split(','))
    val4 = [int(val3[0]), int(val3[1])]
    return val4


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
    n1, n2 = obs.sum(axis=1)
    prop1 = obs[0,0] / n1
    prop2 = obs[1,0] / n2
    delta = prop1 - prop2

    # Wald 95% confidence interval for delta
    se = np.sqrt(prop1*(1 - prop1)/n1 + prop2*(1 - prop2)/n2)
    ci = (delta - 1.96*se, delta + 1.96*se)

    # Yates continuity correction for confidence interval of delta
    correction = 0.5*(1/n1 + 1/n2)
    corrected_ci = (ci[0] - correction, ci[1] + correction)

    return delta, ci, corrected_ci

delta_list = []
ci0_list = []
ci1_list = []

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


plt.fill_between(np.array(df2.index), np.array(ci1_list[0]), np.array(ci0_list[0]), alpha = 0.1, color = 'grey', label = '95% Confidence Interval')
plt.scatter(df2.index, delta_list, c = 'red', label = 'Proportion between Obs and Exp')
plt.xlim([df2.index[0]-1, df2.index[-1]+1])
plt.ylim([-0.5, 0.5])
plt.xlabel('Threshold')
plt.ylabel('Difference in Proportion')
plt.title('Deviation from expected balanced classes')
plt.legend()
plt.show()
plt.savefig('D:/F31/Results/SVM_Freesurfer/27_updated/age_50percent.png')