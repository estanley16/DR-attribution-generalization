#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 14:06:31 2021

@author: emmastanley
"""
import tensorflow as tf 
import numpy as np 
tf.random.set_seed(1234)
np.random.seed(1234)

import argparse 
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import Precision, Recall, AUC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score, classification_report
from tensorflow.keras.applications.inception_v3 import preprocess_input



#input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, help='output prefix to store plots')
parser.add_argument('--modelname', type=str, help='name of model file to evaluate')
args = parser.parse_args()


def ConfusionMatrix(y_true, y_pred):
   
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None)
    disp = disp.plot(include_values=True,cmap='Blues', ax=None, xticks_rotation='horizontal')
    plt.title('Confusion Matrix')
    cmname = plots_dir + args.output + '_cm.png'
    plt.savefig(cmname)

    cm2 = confusion_matrix(y_true, y_pred, normalize='true')
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=None)
    disp2 = disp2.plot(include_values=True,cmap='Blues', ax=None, xticks_rotation='horizontal', values_format= '.0%')
    plt.title('Confusion Matrix (Normalized by True Label)')
    cmname2 = plots_dir + args.output + '_cm_norm.png'
    plt.savefig(cmname2)
    
    return cm


data_dir = '/home/estanley/scratch/DR/data/'
plots_dir = '/home/estanley/scratch/DR/plots/'
model_dir = '/home/estanley/scratch/DR/models/'


test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_it = test_datagen.flow_from_directory(data_dir+'test_eyepacs/',
                                      class_mode='categorical',
                                      batch_size= 1,
                                      color_mode='rgb',
                                      shuffle=False)

model = tf.keras.models.load_model(model_dir + args.modelname)

test_steps = test_it.samples//test_it.batch_size

#get list of filenames the order that datagenerator retrieves them (to match y true, y pred arrays)
filenames = []
for file in test_it.filenames:
    print(file)
    filenames.append(file)
    


import pickle

with open("/home/estanley/scratch/DR/testfilenames_generatorOrder.txt", "wb") as fp:   #Pickling
    pickle.dump(filenames, fp)

    
loss, acc, precision, recall, auc = model.evaluate(test_it, steps=test_steps, verbose=1)

print('Test accuracy {}'.format(acc))
print('Test precision: {}'.format(precision))
print('Test recall: {}'.format(recall))
print('Test AUC: {}'.format(auc))


y_pred_raw = model.predict(test_it, steps=test_steps, verbose=1)
print('Model output: {}'.format(y_pred_raw))

#convert multiclass output to single array
y_pred = np.argmax(y_pred_raw, axis=1).reshape(-1,1)
print('Predictions: {}'.format(y_pred))

#get true labels
y_true_raw = test_it.classes
y_true = np.array(y_true_raw).reshape(-1,1)[:len(y_pred)]
print('True Values: {}'.format(y_true)) 

cm = ConfusionMatrix(y_true, y_pred)
print(cm)

y_true = y_true.reshape(-1)
y_pred = y_pred.reshape(-1)



kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
print(kappa)


#save out y pred and y true arrays 
np.save('/home/estanley/scratch/DR/y_true_' + args.modelname + '.npy', y_true)
np.save('/home/estanley/scratch/DR/y_pred_' + args.modelname + '.npy', y_pred)