# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 12:51:39 2018

@author: edmondsd
Class for running SVM instances with a dataset
"""
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.model_selection import KFold

'''
SVM
'''

def Randomize_groups(df, n):
    df_w = df[df['study_group'] == 1]
    df_c = df[df['study_group'] == 0]
    df_w_sample = df_w.sample(n)
    df_c_sample = df_c.sample(n)
    df_sample = pd.concat([df_w_sample, df_c_sample])
    return df_sample

def full_compare_x(x0, x1):
    result = 0
    if x0 > x1:
        result = 0
    if x1 > x0:
        result = 1    
    return result

def bagging_accuracy(df):   #Not set up for fold changes yet
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
        result = full_compare_x(x0, x1)
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

def accuracy_calc(df, val):
    CORRECT = 0
    WRONG = 0
    for index, row in df.iterrows():
        val2 = row[7]
        if len(row[val]) > 1: #for kfolds
            for num in row[val]:
                if num == val2[num]:
                    CORRECT += 1
                else:
                    WRONG += 1
        elif len(row[val]) <= 1: #for leave-one-out
            if row[val] == val2:
                CORRECT += 1
            else:
                WRONG += 1
    accuracy = CORRECT / (CORRECT + WRONG)
    return accuracy

'''
Data Import
'''
def LoadStatsCSV(filename, index):
    '''
    Imports stats CSV. filename: csv location. Index: How to set index in the file
    Returns CSV file ready for use in ML
    '''
    df = pd.read_csv(filename)
    df.set_index(index, inplace = True)
    df2 = df.apply(lambda x: x.fillna(x.mean()),axis=0)
    return df2

filename = 'D:/F31/DATA/R1/R1_ninety_pca.csv'
df = LoadStatsCSV(filename, 'Code')

'''Create folds'''

NFOLDS = 60 #leave-one-out should be 2*NSUBS
NSUBS = 30
COUNT = 0
df_sample = Randomize_groups(df, NSUBS) #pulls 20 welders/controls into 1 DF and shuffles
df_sample = df_sample.reset_index(drop = True)

df_sample = df_sample[['0','1','2','3','4','5','6','7','8','9','study_group']] #Select columns
#prediction = pd.DataFrame(index=range(2*NSUBS))
prediction = {}
accuracy = {}
val = 'pca'

'''Split Data'''
kd = KFold(n_splits = NFOLDS) #kfold generator, creates NFOLDS 
for i in kd.split(df_sample): #iterates with splits
    train_rows = i[0]
    test_rows = i[1]
    Xtrain = df_sample.iloc[train_rows]
    Xtest = df_sample.iloc[test_rows]
    ytrain = df_sample.iloc[train_rows]
    ytest = df_sample.iloc[test_rows]
    
    '''SVM'''
    test_accuracy = {}
    test_predict = {}
    clf = svm.LinearSVC()
    clf.fit(Xtrain[['0','1','2','3','4','5','6','7','8','9']], Xtrain['study_group'])
    test_accuracy[val] = clf.score(Xtest[['0','1','2','3','4','5','6','7','8','9']],Xtest['study_group'])
    test_predict[val] = clf.predict(Xtest[['0','1','2','3','4','5','6','7','8','9']]) 
    test_predict['study_group'] = np.array(Xtest['study_group'])
    prediction[COUNT] = test_predict
    accuracy[COUNT] = test_accuracy
    COUNT += 1
    print(str(COUNT)+' of '+str(NFOLDS)+' leave-one-out CV folds complete')
    
'''Create DataFrames'''
predict_df = pd.DataFrame.from_dict(prediction)
predict_df = predict_df.transpose()

'''Calculate Accuracy of SVM Model'''
CORRECT = 0
WRONG = 0
for index, row in predict_df.iterrows():
    val2 = row[1]
    if row[val] == val2:
        CORRECT += 1
    else:
        WRONG += 1
accuracy = CORRECT / (CORRECT + WRONG)

print("SVM Accuracy: "+ str(accuracy))
        