# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 15:19:25 2018

@author: edmondsd
"""
import pandas as pd

def LoadStatsCSV(filename, index):
    '''
    Imports stats CSV. filename: csv location. Index: How to set index in the file
    Returns CSV file ready for use in ML
    '''
    df = pd.read_csv(filename)
    df.set_index(index, inplace = True)
    df2 = df.apply(lambda x: x.fillna(x.mean()),axis=0)
    return df2
