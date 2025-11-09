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
#%%
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
    df_target = df_sample[TargetList]
    return df_sample, df_target

def RemoveNA(df):
    df.replace(r'\s+', np.nan, regex=True)
    imp = Imputer()
    imp.fit(df)
    df2 = imp.transform(df)
    return df2

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

def ImportData(STAT, TargetList):
    if STAT == 'skew':
        df = pd.read_csv(DFskew, index_col = 'Code')
        return df
    elif STAT == 'tmean':
        df = pd.read_csv(DFmean, index_col = 'Code')
        return df
    elif STAT == 'tvar':
        df = pd.read_csv(DFvar, index_col = 'Code')
        return df
    elif STAT == 'tmax':
        df = pd.read_csv(DFmax, index_col = 'Code')
        return df
    elif STAT == 'tmin':
        df = pd.read_csv(DFmin, index_col = 'Code')
        return df
    
def Classifier(df, NFOLDS, STAT, column):
    prediction = {}
    COUNT = 0
    kd = KFold(n_splits = NFOLDS) #kfold generator, creates NFOLDS 
    for i in kd.split(df): #iterates with splits
        train_rows = i[0]
        test_rows = i[1]
        Xtrain = df.iloc[train_rows]
        Xtest = df.iloc[test_rows]
        ytrain = df[column].iloc[train_rows]
        ytest = df[column].iloc[test_rows]
    
        '''SVM'''
        test_predict = {}
        clf = svm.LinearSVC(class_weight = 'balanced')
        clf.fit(Xtrain.drop(column, axis = 1), ytrain)
        test_predict[STAT] = clf.predict(Xtest.drop(column, axis = 1))
        test_predict[column] = np.array(ytest)
        prediction[COUNT] = test_predict
        COUNT += 1
#    result = pd.DataFrame.from_dict(OrderedDict(prediction))
#    result = result.transpose()
    return prediction

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
    return df2
#%% Variables for running
    
TARGET = 'study_group'
KERNEL = 'linear'
DFskew = 'D:/F31/DATA/R1/Freesurfer/R1_skew2.csv'
DFmean = 'D:/F31/DATA/R1/Freesurfer/R1_tmean.csv'
DFvar = 'D:/F31/DATA/R1/Freesurfer/R1_tvar.csv'
DFmax = 'D:/F31/DATA/R1/Freesurfer/R1_tmax.csv'
DFmin = 'D:/F31/DATA/R1/Freesurfer/R1_tmin.csv'
TargetList = ['study_group', 'excess_mn', 'air_mn', 'mn_twa']
RANDOMSTATE = 33
NFOLDS = 88 #leave-one-out should be 2*NSUBS
NSUBS = 44

THRESHVECTOR = [1]
#THRESHVECTOR = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
#THRESHVECTOR = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 5, 10]
newdf = pd.DataFrame()
TARGETVECTOR = ['tmean', 'tvar', 'skew', 'tmin', 'tmax']
#storage = pd.DataFrame()

#%%
final = {}
for j in TARGETVECTOR:
    df = ImportData(j, TargetList) 
    df_sample = df.sample(NFOLDS, random_state = RANDOMSTATE)
    df_target_original = df_sample[TargetList]
    df = Standardize(df_sample)
 #   subdf = pd.DataFrame()
    subdict = {}
    storage = {}
    print("Calculating accuracy for "+j)
    for i in THRESHVECTOR:    
        
        df_target = CreateGroups(df_target_original, i, TARGET)
        df = DimensionReduction(df)
        df.set_index(df_target.index, inplace = True)
        df = pd.concat([df, df_target], axis = 1)
#        result = Classifier(df, NFOLDS, j, 'new_target')
        subdict[i] = Classifier(df, NFOLDS, j, 'new_target')
        prediction_vector = []
        target_vector = []
        for val in range(NFOLDS):
            prediction_vector = np.concatenate((prediction_vector, subdict[i][val][j]))
            target_vector = np.concatenate((target_vector, subdict[i][val]['new_target']))
        storage[i] = [prediction_vector, target_vector]
    final[j] = storage
"""
Final contained as [statistic][threshold][0 = class prediction, 1 = class]
"""
#        print("Threshold: "+ str(i) + " with accuracy: "+ str(CalcAccuracy(result, 'new_target')))
 #       subdf = pd.concat([subdf, result], axis = 1)
#    newdf = pd.concat([newdf, subdf], axis = 1)

#pd.DataFrame.to_csv(newdf, 'D:/F31/DATA/R1/freesurfer_results.csv')
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
"""
1. Calculate individual accuracies for each threshold
2. Calculate accuracies across all stats for each threshold
"""
ar_list = []
bagging_list = []
for stat in final.keys():
    for threshold in final[stat].keys():
        df = pd.DataFrame(final[stat][threshold]).T
        ar = [stat, threshold, CalcAccuracy(df)]
        ar_list.append(ar)
        print('Accuracy for '+stat+' at '+str(threshold)+' is: '+str(CalcAccuracy(df)))


threshold_dict = {}
stat_dict = {}

finaldf = pd.DataFrame()
for stat in final.keys():
    middf = pd.DataFrame()
    targetdf = pd.DataFrame()
    for threshold in final[stat].keys():
        df = pd.DataFrame.from_dict(final[stat][threshold]).T
        df.columns = [stat, 'target']
        df['threshold'] = threshold
        df.set_index(['threshold'], inplace = True)
        middf = pd.concat([middf, df], axis = 0).drop(['target'], axis = 1)
        targetdf = pd.concat([targetdf, df], axis = 0).drop([stat], axis = 1)
    finaldf = pd.concat([finaldf, middf], axis = 1)
finaldf = pd.concat([finaldf, targetdf], axis = 1)
    
"""
Loop through and create separate spreadsheet for each threshold w/ column for each variable & 1 column for target
"""        

br_list = []
for val in finaldf.index.unique():
    df = finaldf.loc[val]
    br = [val, Bagging(df)]
    br_list.append(br)
    print("Threshold: "+str(val)+ ' Accuracy = '+ str(Bagging(df)))
    


#%%
bagging_results = pd.DataFrame(np.array(br_list).reshape(len(br_list),2), columns = ['threshold','accuracy'])
bagging_results['combined'] = 'combined'
bagging_results.set_index('combined', inplace = True)
ind_results = pd.DataFrame(np.array(ar_list).reshape(len(ar_list), 3), columns = ['stat', 'threshold', 'accuracy'])
ind_results.set_index('stat', inplace=True)
ind_results['threshold'] = pd.to_numeric(ind_results['threshold'])
ind_results['accuracy'] = pd.to_numeric(ind_results['accuracy'])

export_results = pd.concat([bagging_results, ind_results], axis = 0)
outfile = 'D:/F31/Results/SVM_Freesurfer/'+ TARGET + '_' + str(RANDOMSTATE) + '_fs.csv'
export_results.to_csv(outfile)

tmax = ind_results.loc['tmax']
tmin = ind_results.loc['tmin']
tvar = ind_results.loc['tvar']
tmean = ind_results.loc['tmean']
skew = ind_results.loc['skew']

plt.plot('threshold', 'accuracy', data = bagging_results)
plt.plot('threshold', 'accuracy', data = tmax)
plt.plot('threshold', 'accuracy', data = tmin)
plt.plot('threshold', 'accuracy', data = tvar)
plt.plot('threshold', 'accuracy', data = tmean)
plt.plot('threshold', 'accuracy', data = skew)
plt.legend(['Combined', 'Maximum', 'Minimum', 'Variance', 'Mean', 'Skew'])
plt.title('Accuracy Across Threshold Levels for Air Mn Concentration')
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.ylim([0.3,.9])
#plt.xscale('log')
plt.savefig('D:/F31/Results/SVM_Freesurfer/'+TARGET+'_'+str(RANDOMSTATE)+'_fs.png')
plt.show()
