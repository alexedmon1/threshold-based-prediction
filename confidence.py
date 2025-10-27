# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 20:33:34 2019

@author: edmondsd

Confidence
3 = Borderline
4 = Confident
5 = Very Confident
"""

import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('D:/F31/Results/SVM_Freesurfer/ResultsFixed/th_gaba_2_predictions.csv')
df.set_index('threshold',inplace = True)

     
d = {}
for thresh in df.index.unique():
    sub = df.loc[thresh]
    results = []
    for row in sub.iterrows():
        x = 0
        y = 0
        stats = row[-1]
        for val in stats[0:5]:
            if val == stats[5]:
                x += 1
            elif val != stats[5]:
                y += 1
        results.append(x)
        d[thresh] = Counter(results)
                  
"""
Logic -
1) Identify the real
2) Does majority rule pick it correctly?
    if yes: determine confidence
    if no: 'incorrect'
"""

fig, ax  = plt.subplots(figsize = [10,6])
N = len(d.keys())
ind = np.arange(N)
width = 0.2

borderline = []
confident = []
very_confident = []
incorrect = []

for key, value in d.items():
    borderline.append(value[3])
    confident.append(value[4])
    very_confident.append(value[5])
    incorrect.append(value[0] + value[1] + value[2])


ax.set_xticklabels(d.keys())
ax.set_xticks(ind + width)
plot_very_confident = ax.bar(ind-width, very_confident, width, bottom = 0, align = 'center', color = 'white', edgecolor = 'black')
plot_confident = ax.bar(ind, confident, width, bottom = 0, align = 'center', color = 'lightgrey', edgecolor = 'black')
plot_borderline = ax.bar(ind+width, borderline, width, bottom = 0,align = 'center',  color = 'grey', edgecolor = 'black')
plot_incorrect = ax.bar(ind+2*width, incorrect, width, bottom = 0,align = 'center',  color = 'black', edgecolor = 'black')
plt.legend((plot_very_confident[0], plot_confident[0], plot_borderline[0], plot_incorrect[0]),('High (5)', 'Medium (4)', 'Low (3)', 'Incorrect (<=2)'))    
plt.title('Confidence levels of predictions for class (Thalamic GABA)')
plt.ylabel('Counts')
plt.xlabel('Threshold (mM)')

plt.savefig('D:/F31/Results/SVM_Freesurfer/ResultsFixed/th_gaba_confidence.png', dpi = 600)
plt.show()