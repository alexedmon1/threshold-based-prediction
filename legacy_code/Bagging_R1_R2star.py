# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 14:49:07 2018

@author: edmondsd
Import R1 and R2* predictions file

"""
#%%
import pandas as pd

mn = pd.read_csv('D:/F31/Results/SVM_Freesurfer/study_group_33_predictions.csv', index_col = 'threshold')
fe = pd.read_csv('D:/F31/Results/SVM_fs_r2star/study_group_33_predictions.csv', index_col = 'threshold')

mn = mn.drop('target', axis = 1)
combined = pd.concat([mn, fe], axis = 1)

#%%
def CalcAccuracy(df):
   CORRECT = 0
   WRONG = 0
   for index, row in df.iterrows():
       if row[0] == row[1]:
           CORRECT += 1
       else:
           WRONG += 1
   accuracy = CORRECT / (CORRECT + WRONG)
   return accuracy

def CompareX(x0, x1):
    result = 0
    if x0 =< x1:
        result = 0
    if x1 > x0:
        result = 1    
    return result

def BaggingBoth(df):   #Not set up for fold changes yet
    bagging_df = pd.DataFrame()
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
#%%
for i in combined.index.unique():
    print('Accuracy at Threshold: '+str(i)+' is '+str(BaggingBoth(combined.loc[i])))    
