#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 14:42:29 2021

@author: emmastanley
"""

import tensorflow as tf 
import numpy as np 
tf.random.set_seed(1234)
np.random.seed(1234)

import argparse 
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import Precision, Recall, AUC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, average_precision_score
from tensorflow.keras.applications.inception_v3 import preprocess_input

from sklearn.utils import class_weight

#input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, help='output prefix to store weights and log files')
parser.add_argument('--epochs', type=int, help='number of traning epochs to run')
parser.add_argument('--l2', type=float, help='l2 regularizer for last dense layer')
# parser.add_argument('--layername', type=str, help='name of inceptionv3 to start unfreeze model params at')
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


data_dir = '/home/estanley/scratch/DR/data/'
plots_dir = '/home/estanley/scratch/DR/plots/'
model_dir = '/home/estanley/scratch/DR/models/'
csv_dir = '/home/estanley/scratch/DR/history/'


train_datagen = ImageDataGenerator(rotation_range=360,
                                   shear_range=15.0,
                                   width_shift_range=[-10,10],
                                   height_shift_range=[-10,10],
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator()

#define iterators for loading images from datasets
train_it = train_datagen.flow_from_directory(data_dir+'train_eyepacs/', 
                                       class_mode='categorical', 
                                       batch_size= 12,
                                       color_mode='rgb',
                                       shuffle=True)

val_it = test_datagen.flow_from_directory(data_dir+'val_eyepacs/', 
                                     class_mode='categorical',
                                     batch_size= 12,
                                     color_mode='rgb',
                                     shuffle=True)

test_it = test_datagen.flow_from_directory(data_dir+'test_eyepacs/',
                                      class_mode='categorical',
                                      batch_size= 12,
                                      color_mode='rgb',
                                      shuffle=False)


#confirm iterator works
batchX, batchy = train_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

train_steps_per_epoch = train_it.samples//train_it.batch_size
val_steps_per_epoch = val_it.samples//val_it.batch_size

weights = class_weight.compute_class_weight(
               'balanced',
                np.unique(train_it.classes), 
                train_it.classes)
class_weights = {i : weights[i] for i in range(5)}


#create the model 
#https://keras.io/api/applications/
base_model = tf.keras.models.load_model('/home/estanley/scratch/DR/inceptionv3')
# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2=args.l2), kernel_initializer=tf.keras.initializers.HeNormal())(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(5, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False
    
# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

model.summary() #check trainable layers

# train the model on the new data for a few epochs
history_init = model.fit(train_it, 
                    steps_per_epoch=train_steps_per_epoch,
                    epochs=5,
                    class_weight=class_weights,
                    # validation_data=val_it,
                    # validation_steps=val_steps_per_epoch,
                    verbose=2)




#unfreeze whole models 
for layer in model.layers:
    layer.trainable = True
    
    
# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['acc', Precision(), Recall(), AUC()])


model.summary() #check trainable layers 

fname = model_dir + args.output + '_weights'
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(fname, save_best_only=True)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
csv_logger_cb = tf.keras.callbacks.CSVLogger(csv_dir + args.output + '.csv')

#fit model
history = model.fit(train_it, 
                    steps_per_epoch=train_steps_per_epoch,
                    epochs=args.epochs,
                    class_weight=class_weights,
                    validation_data=val_it,
                    validation_steps=val_steps_per_epoch,
                    callbacks=[checkpoint_cb, early_stopping_cb, csv_logger_cb],
                    verbose=2)



#plot accuracy 
pname = plots_dir + args.output + '_accuracy.png'
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc = 'upper left')
plt.savefig(pname)
plt.clf()

#plot loss
pname = plots_dir + args.output + '_loss.png'
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(pname)
plt.clf()


test_steps = test_it.samples//test_it.batch_size

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

ConfusionMatrix(y_true, y_pred)
