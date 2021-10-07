#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 14:30:12 2021

@author: emmastanley
"""
import numpy as np 
import pandas as pd 
import pickle


load_dir = '/Users/emmastanley/Documents/BME/Research/DR/TCAV/eyepacs_test_data/'
modelname = '13_inception_full_weights'
dataset = 'eyepacs'

#load true and predicted arrays from models
y_true = np.load(load_dir + 'y_true_' + modelname + '_' + dataset + '.npy')
y_pred = np.load(load_dir + 'y_pred_' + modelname + '_' + dataset + '.npy')


#find indices of correct and incorrect classifications
T_idx = np.where((y_pred == y_true))[0]
F_idx = np.where((y_pred != y_true))[0]



#open list of files in order that data generator reads them
with open(load_dir + 'testfilenames_generatorOrder_' + dataset + '.txt', "rb") as fp:   # Unpickling
   # filepaths_list = pickle.load(fp)
   filenames_list = pickle.load(fp)
   
filenames_list = np.array(filenames_list[:len(y_true)]) #to correctly sized array

#names of correct (T) and incorrect (F) classified files
T_filenames = filenames_list[T_idx]
F_filenames = filenames_list[F_idx]

T_filenames = T_filenames.reshape(-1,1)
F_filenames = F_filenames.reshape(-1,1)


#save out file paths as .txt for linux to read
with open(load_dir + modelname + '_' + dataset + "_correctclass.txt", 'w') as file1:
        for row1 in T_filenames:
            s1 = " ".join(map(str, row1))
            file1.write(s1+'\n')
            
with open(load_dir + modelname + '_' + dataset + "_incorrectclass.txt", 'w') as file2:
        for row2 in F_filenames:
            s2 = " ".join(map(str, row2))
            file2.write(s2+'\n')
                        