# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 10:08:15 2018

@author: edmondsd
Final SVM worksheet
For classifying 2 groups using R1

1. Set threshold
2. Set target
3. Set kernel
4. Import skew, mean, and ninety
5. Run PCA & transform
6. SVM 
7. Bagging
8. Report individual accuracies & bagged accuracies 
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import KFold

def RandomizeGroups(df, n, column):
    df_1 = df[df[column] == 1]
    df_0 = df[df[column] == 0]
    df_1_sample = df_1.sample(n)
    df_0_sample = df_0.sample(n)
    df_sample = pd.concat([df_1_sample, df_0_sample])
    return df_sample

def RemoveNA(df):
    df.replace(r'\s+', np.nan, regex=True)
    imp = Imputer()
    imp.fit(df)
    df2 = imp.transform(df)
    return df2

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

def CalcAccuracy(df, column):
    CORRECT = 0
    WRONG = 0
    for index, row in df.iterrows():
        val2 = row[1]
        if row[column] == val2:
            CORRECT += 1
        else:
            WRONG += 1
    accuracy = CORRECT / (CORRECT + WRONG)
    return accuracy

def Standardize(df):
    sc = StandardScaler()
    df_std = sc.fit_transform(RemoveNA(df.drop(TargetList, axis = 1)))
    return df_std

def DimensionReduction(df):
    pca = PCA()
    pca.fit(df)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum >= 0.90) + 1
    pca = PCA(n_components = d)
    pca.fit(df)
    new_array = np.dot(pca.components_, df.T)
    new_df = pd.DataFrame(new_array).transpose()
    return new_df

def TargetDF(STAT, TargetList):
    if STAT == 'skew':
        df = pd.read_csv(DFskew, index_col = 'Code')
        df_target = df[TargetList]
        df = Standardize(df)
        return df, df_target
    elif STAT == 'mean':
        df = pd.read_csv(DFmean, index_col = 'Code')
        df_target = df[TargetList]
        df = Standardize(df)
        return df, df_target
    elif STAT == 'ninety':
        df = pd.read_csv(DFninety, index_col = 'Code')
        df_target = df[TargetList]
        df = Standardize(df)
        return df, df_target

def Classifier(df, NFOLDS, column):
    prediction = {}
    COUNT = 0
    kd = KFold(n_splits = NFOLDS) #kfold generator, creates NFOLDS 
    for i in kd.split(df_sample): #iterates with splits
        train_rows = i[0]
        test_rows = i[1]
        Xtrain = df_sample.iloc[train_rows]
        Xtest = df_sample.iloc[test_rows]
        ytrain = df_sample[column].iloc[train_rows]
        ytest = df_sample[column].iloc[test_rows]
    
        '''SVM'''
        test_predict = {}
        clf = svm.LinearSVC()
        clf.fit(Xtrain.drop(column, axis = 1), ytrain)
        test_predict[STAT] = clf.predict(Xtest.drop(column, axis = 1))
        test_predict[column] = np.array(ytest)
        prediction[COUNT] = test_predict
        COUNT += 1
    predict_df = pd.DataFrame.from_dict(prediction)
    predict_df = predict_df.transpose()
    return predict_df

def CreateGroups(df, THRESH, column):
    """
    Create 2 groups: Low (<0.1), High (>=0.1)
    """
    d = {}
    v = df[column]
    for subj in v.index:
        if v.loc[subj] < THRESH:
            d[subj] = 0
        else:
            d[subj] = 1
    df2 = pd.DataFrame.from_dict(d, orient = 'index')
    df2.columns = ['new_target']
    df = pd.concat([df, df2], axis = 1)
    return df

STAT = 'skew'
TARGET = 'excess_mn'
KERNEL = 'linear'
DFskew = 'D:/F31/DATA/R1/Freesurfer/R1_skew.csv'
DFmean = 'D:/F31/DATA/R1/Freesurfer/R1_mean.csv'
DFninety = 'D:/F31/DATA/R1/Freesurfer/R1_ninety.csv'
TargetList = ['study_group', 'excess_mn', 'air_mn', 'mn_twa']
NFOLDS = 60 #leave-one-out should be 2*NSUBS
NSUBS = 30


#THRESHVECTOR = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
THRESHVECTOR = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2]
for i in THRESHVECTOR:
    df, df_target = TargetDF(STAT, TargetList)   
    df_target = CreateGroups(df_target, i, TARGET)
    df = DimensionReduction(df)
    df.set_index(df_target.index, inplace = True)
    df = pd.concat([df, df_target], axis = 1)
    df_sample = RandomizeGroups(df, NSUBS, 'new_target')
    df_sample = df_sample.reset_index(drop = True).drop(TargetList, axis = 1)
    result = Classifier(df_sample, NFOLDS, 'new_target')
    print("Threshold: "+ str(i) + " with accuracy: "+ str(CalcAccuracy(result, 'new_target')))
