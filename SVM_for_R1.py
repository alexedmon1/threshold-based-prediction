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
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

def RandomizeGroups(df, n, column):
    df_1 = df[df[column] == 1]
    df_0 = df[df[column] == 0]
    df_1_sample = df_1.sample(n, sort = True)
    df_0_sample = df_0.sample(n, sort = True)
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

def ImportData(STAT, TargetList):
    if STAT == 'skew':
        df = pd.read_csv(DFskew, index_col = 'Code')
        return df
    elif STAT == 'median':
        df = pd.read_csv(DFmedian, index_col = 'Code')
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
    elif STAT == 'ten':
        df = pd.read_csv(DFten, index_col = 'Code')
        return df
    elif STAT == 'ninety':
        df = pd.read_csv(DFninety, index_col = 'Code')
        return df
    
def CreateGroups(df, THRESH, column):
    """
    Create 2 groups: Low (<0.1), High (>=0.1)
    Counts n in each category
    """
    d = {}
    v = df[column]
    COUNT0 = 0
    COUNT1 = 0
    for subj in v.index:
        if v.loc[subj] <= THRESH:
            d[subj] = 0
            COUNT0 += 1
        else:
            d[subj] = 1
            COUNT1 += 1
    df2 = pd.DataFrame.from_dict(d, orient = 'index')
    df2.columns = ['new_target']
    df = pd.concat([df, df2], axis = 1)
    counts = [COUNT0, COUNT1]
    return df2, counts

def PlotPCA(EVR, TARGET, STAT):
    plt.xlim([0,len(EVR)+1])
    plt.title('PCA of '+STAT+' R1 in ROIs')
    plt.ylabel('Percentage of Explained variance')
    plt.xlabel('Eigenvector')
    plt.plot(np.cumsum(EVR)) #pca.explained_variance_ratio
    plt.plot(EVR) #pca.explained_variance_ratio
    plt.savefig('D:/F31/Results/SVM_Freesurfer/PCA/'+TARGET+'_'+str(j)+'.png')
    plt.close()

def DimensionReduction(df):
    pca = PCA()
    pca.fit(df)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum >= 0.90) + 1
    pca = PCA(n_components = d)
    pca.fit(df)
    new_array = np.dot(pca.components_, df.T)
    new_df = pd.DataFrame(new_array).transpose()
    return new_df, pca.explained_variance_ratio_, pca.components_

def ComponentsPCA(COMPONENTS, COLUMNS):
    c1 = pd.DataFrame(COMPONENTS[0])
    c2 = pd.DataFrame(COMPONENTS[1])
    c3 = pd.DataFrame(COMPONENTS[2])
    dff = pd.concat([c1, c2, c3], axis = 1)
    dff['ROI'] = COLUMNS
    return dff

def KMeansClusterPlot(df, TARGET, j, i):
    pc1 = df[0].values
    pc2 = df[1].values
    pc3 = df[2].values
    X = np.array(list(zip(pc1, pc2, pc3)))
    #plst.scatter(pc1,pc2)
    # Number of clusters
    kmeans = KMeans(n_clusters=2)
    # Fitting the input data
    kmeans = kmeans.fit(X)
    # Getting the cluster labels
    labels = kmeans.predict(X)
    # Centroid values
    #centroids = kmeans.cluster_centers_
    C = kmeans.cluster_centers_
    
    LABEL_COLOR_MAP = {0 : 'red',
                       1 : 'blue',
                       2 : 'green'}

    label_color = [LABEL_COLOR_MAP[l] for l in df['new_target']]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('K-means Cluster for '+TARGET+' '+j+' with THRESH = '+str(i))
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=label_color)
    ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='#050505', s=1000)
    ax.view_init(30, 225)
#    ax.legend([Red = 0, Blue = 1])
    plt.savefig('D:/F31/Results/SVM_Freesurfer/KMeansCluster/'+TARGET+'_'+str(j)+'_'+str(i)+'.png')
    plt.close(fig)
    
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
        for val in row[0:len(df.columns)-1]:
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


#%% Variables for running
    
TARGET = 'mn_twa'
KERNEL = 'linear'
DFskew = 'D:/F31/DATA/R1/Freesurfer/R1_skew2.csv'
DFmedian = 'D:/F31/DATA/R1/Freesurfer/R1_median.csv'
DFvar = 'D:/F31/DATA/R1/Freesurfer/R1_tvar.csv'
DFmax = 'D:/F31/DATA/R1/Freesurfer/R1_tmax.csv'
DFmin = 'D:/F31/DATA/R1/Freesurfer/R1_tmin.csv'
DFten = 'D:/F31/DATA/R1/Freesurfer/R1_ten.csv'
DFninety = 'D:/F31/DATA/R1/Freesurfer/R1_ninety.csv'
TargetList = ['study_group', 'excess_mn', 'air_mn', 'mn_twa', 'total_welding_years', 'th_gaba', 'updrs_score', 'toenail_mn', 'mn_diet', 'age', 'total_mn', 'total_mn_90', 'total_mn_365']
RANDOMSTATE = 27
NFOLDS = 89 #leave-one-out should be 2*NSUBS
NSUBS = 44

#THRESHVECTOR = [0]
THRESHVECTOR = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25] # mn_twa (0-0.125)
#THRESHVECTOR = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5] # excess_mn (0-4.089)
#THRESHVECTOR = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] #years (0-13.5)
#THRESHVECTOR = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5] #th_gaba
#THRESHVECTOR = [24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60]
#THRESHVECTOR = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
newdf = pd.DataFrame()
TARGETVECTOR = ['median', 'tvar', 'skew', 'ten', 'ninety']
#storage = pd.DataFrame()

#%%
"""
Final contained as [statistic][threshold][0 = predicted class, 1 = class]
"""
final = {}
counting_final = {}
for j in TARGETVECTOR:
    df = ImportData(j, TargetList) 
    df_sample = df.sample(NFOLDS, random_state = RANDOMSTATE)
    df_target_original = df_sample[TargetList]
    dfstd = Standardize(df_sample)
 #   subdf = pd.DataFrame()
    subdict = {}
    storage = {}
    counting = {}
    print("Calculating accuracy for "+j)
    for i in THRESHVECTOR:         
        df_target, counts = CreateGroups(df_target_original, i, TARGET)
        dfDR, EVR, COMPONENTS = DimensionReduction(dfstd)
        dfDR.set_index(df_target.index, inplace = True)
        df2 = pd.concat([dfDR, df_target], axis = 1)
#        result = Classifier(df, NFOLDS, j, 'new_target')
        subdict[i] = Classifier(df2, NFOLDS, j, 'new_target')
        prediction_vector = []
        target_vector = []
        for val in range(NFOLDS):
            prediction_vector = np.concatenate((prediction_vector, subdict[i][val][j]))
            target_vector = np.concatenate((target_vector, subdict[i][val]['new_target']))
        storage[i] = [prediction_vector, target_vector]
        counting[i] = counts
 #       KMeansClusterPlot(df2, TARGET, j, i)
#    PlotPCA(EVR, TARGET, j)
    print(EVR)
#    export_components = ComponentsPCA(COMPONENTS, df_sample.iloc[:,0:192].columns)
#    export_components.columns = ['1', '2', '3', 'ROI']
#    export_components.to_csv('D:/F31/Results/SVM_Freesurfer/PCA/'+TARGET+'_'+str(RANDOMSTATE)+'_'+j+'components.csv')
    final[j] = storage
    counting_final[j] = counting

   
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
    bagdf = finaldf.loc[val]
    br = [val, Bagging(bagdf)]
    br_list.append(br)
    print("Threshold: "+str(val)+ ' Accuracy = '+ str(Bagging(df)))
    
#%%

ns = pd.DataFrame.from_dict(counting_final, orient = 'columns')
nfile = 'D:/F31/Results/SVM_Freesurfer/ResultsFixed/'+TARGET+'_'+str(RANDOMSTATE)+'_ns.csv'
ns.to_csv(nfile)

#%%

ar_file = pd.DataFrame(ar_list)
br_file = pd.DataFrame(br_list)
arname = 'D:/F31/Results/SVM_Freesurfer/ResultsFixed/'+TARGET+'_'+str(RANDOMSTATE)+'_sep_accuracies.csv'
brname = 'D:/F31/Results/SVM_Freesurfer/ResultsFixed/'+TARGET+'_'+str(RANDOMSTATE)+'_bag_accuracies.csv'
ar_file.to_csv(arname)
br_file.to_csv(brname)

#%%
bagging_results = pd.DataFrame(np.array(br_list).reshape(len(br_list),2), columns = ['threshold','accuracy'])
bagging_results['combined'] = 'combined'
bagging_results.set_index('combined', inplace = True)
ind_results = pd.DataFrame(np.array(ar_list).reshape(len(ar_list), 3), columns = ['stat', 'threshold', 'accuracy'])
ind_results.set_index('stat', inplace=True)
ind_results['threshold'] = pd.to_numeric(ind_results['threshold'])
ind_results['accuracy'] = pd.to_numeric(ind_results['accuracy'])

export_results = pd.concat([bagging_results, ind_results], axis = 0)
outfile = 'D:/F31/Results/SVM_Freesurfer/ResultsFixed/'+ TARGET + '_' + str(RANDOMSTATE) + '_fs.csv'
export_results.to_csv(outfile)
predfile = 'D:/F31/Results/SVM_Freesurfer/ResultsFixed/'+TARGET+'_'+str(RANDOMSTATE)+'_predictions.csv'
finaldf.to_csv(predfile)


ninety = ind_results.loc['ninety']
ten = ind_results.loc['ten']
tvar = ind_results.loc['tvar']
median = ind_results.loc['median']
skew = ind_results.loc['skew']

plt.plot('threshold', 'accuracy', data = bagging_results)
plt.plot('threshold', 'accuracy', data = ninety)
plt.plot('threshold', 'accuracy', data = ten)
plt.plot('threshold', 'accuracy', data = tvar)
plt.plot('threshold', 'accuracy', data = median)
plt.plot('threshold', 'accuracy', data = skew)
plt.legend(['Combined', '90 Percentile', '10 Percentile', 'Variance', 'Median', 'Skew'])
plt.title('Accuracy Across Age')
plt.ylabel('Accuracy')
plt.ylim([0.3,1])
plt.xlabel('Threshold (Years)')
#plt.xscale('log')
plt.savefig('D:/F31/Results/SVM_Freesurfer/ResultsFixed/'+TARGET+'_'+str(RANDOMSTATE)+'_fs.png')
plt.show()
