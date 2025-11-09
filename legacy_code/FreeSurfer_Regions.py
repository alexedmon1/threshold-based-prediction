# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 17:22:25 2018

@author: edmondsd
"""

''' Dictionary for regions 
    1) BrainStruct = Structural clustering of regions
    2) BrainFunct = Functional clustering of regions
'''

''' Imports '''
import pandas as pd
import re

''' Any file w/ 192 regions from FreeSurfer '''
def BrainStruct():
    df = pd.read_csv('C:/Users/edmondsd/DATA/mn_pbpk/R1/R1_welder.csv') #List of all ROIs
    labels = [val.strip() for val in df['ROI'].unique()]

    ''' BrainStruct '''
    BrainStruct = {} #create dictionary

    p = re.compile('[ctx+]') #find all w/ cortex
    cortex = []
    for val in labels:
        m = p.match(val)
        if m:
            cortex.append(val)
        else:
            None    

    BrainStruct['Cortex'] = cortex #Maintaining cortex as its own
    BrainStruct['Cerebellum'] = ['Left-Cerebellum-White-Matter', 'Left-Cerebellum-Cortex', 'Right-Cerebellum-White-Matter', 'Left-Cerebellum-Cortex']
    BrainStruct['BasalGanglia'] = ['Left-Thalamus-Proper', 'Left-Caudate', 'Left-Putamen', 'Left-Pallidum', 'Right-Thalamus-Proper', 'Right-Caudate', 'Right-Putamen', 'Right-Pallidum']
    BrainStruct['Ventricles'] = ['Left-Lateral-Ventricle', 'Left-Inf-Lat-Vent', '3rd-Ventricle', '4th-Ventricle', 'CSF', 'Right-Lateral-Ventricle', 'Right-Inf-Lat-Vent']
    BrainStruct['WhiteMatter'] = ['Left-Cerebral-White-Matter', 'Right-Cerebral-White-Matter', 'CC_Posterior', 'CC_Mid_Posterior', 'CC_Central', 'CC_Mid_Anterior', 'CC_Anterior']
    BrainStruct['InnerBrain'] = ['Brain-Stem', 'Left-choroid-plexus', 'Right-choroid-plexus', 'Left-Hippocampus', 'Left-Amygdala', 'Left-VentralDC', 'Left-Accumbens-area', 'Right-Hippocampus', 'Right-Amygdala', 'Right-VentralDC', 'Right-Accumbens-area']
    BrainStruct['Blood'] = ['Left-vessel', 'Right-vessel']

    return BrainStruct