#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 14:46:22 2021

@author: emmastanley
"""

import numpy as np 
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

#%%
#Plot all TCAV results regardless of significance

layers = ['mixed0', 'mixed2', 'mixed4', 'mixed6', 'mixed8', 'mixed10']
sns.set(style="whitegrid")

# classes = ['class0', 'class1', 'class2', 'class3', 'class4']
classes = ['class4']


for classname in classes: 
    data = pd.read_csv('/Users/emmastanley/Documents/BME/Research/DR/TCAV/Results/Exp4/model 14 test/exp4MODEL14_'+ classname+ '_model13full_correct_eyepacs.csv', index_col=0)
    
    concepts = ['hemmorhage_cropped', 'microaneursym_cropped', 'hardexudate_cropped', 'tortuous_cropped', 'softexudate_cropped']
    # concepts = ['hemmorhage_full', 'microaneursym_full', 'hardexudate_full', 'tortuous_full', 'softexudate_full']
    
    for concept in concepts :
        df = data.loc[data['concept'] == concept]
        
        df['TCAV scores'] = df['TCAV scores'].str.split(',') #convert each list of tcav scores into an actual python list object
        
        for index, row in df.iterrows():
            lst = row['TCAV scores']
            new_lst = [s.strip('[]') for s in lst]
            df.at[index, 'TCAV scores'] = new_lst
            
        df_full = df.explode('TCAV scores', ignore_index=True) #expand each list to its own column
        df_full['TCAV scores']=pd.to_numeric(df_full['TCAV scores']) #convert all tcav values to numeric

        sns.boxplot(x='layer', y='TCAV scores', data=df_full).set_title(classname+' '+concept) #plot!!!! finally
        plt.show()
        sns.stripplot(x='layer', y='TCAV scores', data=df_full).set_title(classname+' '+concept)
        plt.show()
    
    
    
#%%    
#plot only layers with significant p-values

layers = ['mixed0', 'mixed2', 'mixed4', 'mixed6', 'mixed8', 'mixed10']
sns.set(style="whitegrid")
 

# classes = ['class0', 'class1', 'class2', 'class3', 'class4']
classes = ['class4']
concepts = ['hemmorhage_cropped', 'microaneursym_cropped', 'hardexudate_cropped', 'tortuous_cropped', 'softexudate_cropped']
for classname in classes: 
    data = pd.read_csv('/Users/emmastanley/Documents/BME/Research/DR/TCAV/Results/Exp4/model test/exp4MODELTEST_'+ classname+ '_model13full_correct_eyepacs.csv', index_col=0)
    
    #drop insignificant rows
    data = data.drop(data[data['p val'] > 0.05].index)
    
    concepts = ['hemmorhage_cropped', 'microaneursym_cropped', 'hardexudate_cropped', 'tortuous_cropped', 'softexudate_cropped']
    # concepts = ['hemmorhage_full', 'microaneursym_full', 'hardexudate_full', 'tortuous_full', 'softexudate_full']
    
    for concept in concepts :
        df = data.loc[data['concept'] == concept]
        
        df['TCAV scores'] = df['TCAV scores'].str.split(',') #convert each list of tcav scores into an actual python list object
        
        for index, row in df.iterrows():
            lst = row['TCAV scores']
            new_lst = [s.strip('[]') for s in lst]
            df.at[index, 'TCAV scores'] = new_lst
            
        df_full = df.explode('TCAV scores', ignore_index=True) #expand each list to its own column
        df_full['TCAV scores']=pd.to_numeric(df_full['TCAV scores']) #convert all tcav values to numeric

        sns.boxplot(x='layer', y='TCAV scores', data=df_full).set_title(classname+' '+concept) #plot!!!! finally
        plt.show()
        sns.stripplot(x='layer', y='TCAV scores', data=df_full).set_title(classname+' '+concept)
        plt.show()
    