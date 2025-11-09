# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 21:54:53 2018

@author: edmondsd
Bagging analysis of freesurfer results
"""
import numpy as np
import pandas as pd

def CompareX(x0, x1):
    result = 0
    if x0 > x1:
        result = 0
    if x1 > x0:
        result = 1    
    return result

def Bagging(df):   #Not set up for fold changes yet
    bagging_df = pd.DataFrame()
    for index, row in df.iterrows(): 
        x0 = 0
        x1 = 0
        result = 0
        for val in row[0:7]:
            if val == 0:
                x0 += 1
            if val == 1:
                x1 += 1
        result = CompareX(x0, x1)
        vector = pd.Series([index, row[7], result])
        bagging_df = pd.concat([bagging_df, vector], axis = 1)

    bag_df = bagging_df.transpose()
    bag_df.columns = ['subject', 'study_group', 'result']

    wrong = 0
    correct = 0
    for index, row in bag_df.iterrows():
        if row[1] != row[2]:
            wrong += 1
        if row[1] == row[2]:
            correct += 1
            
    accuracy = correct / (wrong + correct)
    return accuracy

df = pd.read_csv('D:/F31/DATA/R1/freesurfer_combined.csv', index_col = 'threshold')
df.loc[0.06]

bagging_df = pd.DataFrame()
THRESHOLDVECTOR = [0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12]
for i in THRESHOLDVECTOR:
    threshdf = df.loc[i]
    for index, row in threshdf.iterrows():
        x0 = 0
        x1 = 0
        result = 0
        for val in row[0:4]:
            if val == 0:
                x0 += 1
            if val == 1:
                x1 += 1
        result = CompareX(x0, x1)
        vector = pd.Series([index, row[3], result])
        bagging_df = pd.concat([bagging_df, vector], axis = 1)  

    bag_df = bagging_df.transpose()
    bag_df.columns = ['subject', 'study_group', 'result']

    wrong = 0
    correct = 0
    for index, row in bag_df.iterrows():
        if row[1] != row[2]:
            wrong += 1
        if row[1] == row[2]:
            correct += 1
            
    accuracy = correct / (wrong + correct)
    print('Accuracy = ' + str(accuracy) + ' for Threshold = '+ str(i))