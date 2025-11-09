# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 10:26:40 2019

@author: edmondsd
"""

"""
ROC Curve Analysis of SVM Models
1) Import predictions for a target
2) Split into thresholds and calculate tpr & fpr for each one
3) Plot 
4) Calculate best threshold & AUC
"""

#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
        for val in row[0:len(df.columns)-1]:
            if val == 0:
                x0 += 1
            if val == 1:
                x1 += 1
        result = CompareX(x0, x1)
        vector = pd.Series([index, row[-1], result])
        bagging_df = pd.concat([bagging_df, vector], axis = 1)

    bag_df = bagging_df.transpose()
    bag_df.columns = ['threshold', 'true', 'predicted']
    return bag_df

def CalcRates(df):
    fp = 0
    tp = 0
    fn = 0
    tn = 0
    for row in df.iterrows():
        vals = row[-1]
        """ -1 = predicted, -2 = true """
        if vals[-1] == vals[-2] and vals[-2] == 1:
            tp += 1
        elif vals[-1] == vals[-2] and vals[-2] == 0:
            tn += 1
        elif vals[-1] != vals[-2] and vals[-2] == 1:
            fn += 1
        elif vals[-1] != vals[-2] and vals[-2] == 0:
            fp += 1
    return tp, tn, fn, fp

#def CalcAUC():

def CalcThreshold(fpr, tpr, threshold): #fpr = Specificity, tpr = sensitivity
    diff = tpr*(1-fpr)
    maximum = np.max(diff)
    ind = np.where(diff == maximum)
    return fpr[ind], tpr[ind], threshold[ind]

#%%
variable = 'updrs_score'
df = pd.read_csv('D:/F31/Results/SVM_Freesurfer/27_updated/'+variable+'_27_predictions.csv')
df.set_index(['threshold'], inplace = True)


#%%
tpr = []
fpr = []
recall = []
precision = []
thresh = []
accuracy = []
F1 = []
d = {}
for threshold in df.index.unique():
    dfthresh = df.loc[threshold]
    """ Calculate Bagging Prediction """
    bagdf = Bagging(dfthresh)
    """ calculate tpr """
    tp, tn, fn, fp = CalcRates(bagdf)
    tpr.append(tp/(tp + fn))
    fpr.append(fp/(fp + tn))
    prec = (tp / (tp + fp))
    rec = (tp / (tp + fn))
    f1 = 2*((prec*rec)/(prec + rec))
    recall.append(rec)
    precision.append(prec)
    accuracy.append((tp + tn) / (tp + tn + fp + fn))
    F1.append(f1)
    thresh.append(threshold)
    
#%%
def MakeDF(list1, list2):
    df = pd.DataFrame(list2, list1)
    return df
    
df_recall = MakeDF(thresh, recall)
df_precision = MakeDF(thresh, precision)
df_accuracy = MakeDF(thresh, accuracy)
df_F1 = MakeDF(thresh, F1)

df_total = pd.DataFrame()
df_total = pd.concat([df_total, df_recall, df_precision, df_accuracy, df_F1], axis = 1)
df_total.columns = ['Recall', 'Precision', 'Accuracy', 'F1']

df_total.to_csv('D:/F31/Results/SVM_Freesurfer/27_Updated/forthesis/'+variable+'_scoring.csv')

#%%
plt.plot(thresh,accuracy,label = "Accuracy")
plt.plot(thresh,recall,label = "Recall")
plt.title('UPDRS Score')
plt.xlabel('Threshold (Score)')
plt.ylabel('Percent')
plt.ylim([0, 1.05])
plt.legend()
plt.savefig('D:/F31/Results/SVM_Freesurfer/27_Updated/forthesis/'+variable+'_scoring.png')
plt.show()


