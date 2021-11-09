#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 10:42:10 2021

@author: emmastanley
"""
import numpy as np 
import pandas as pd 
import os 
import argparse 
import tensorflow as tf 
import matplotlib.pyplot as plt
import tf_keras_vis
from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras import backend as K
from tf_keras_vis.saliency import Saliency
from matplotlib import cm
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils import normalize

file_dir = '/Users/emmastanley/Documents/BME/Research/DR/saliency/model13full/correctly_classified_eyepacs/'
# file_directory = '/home/estanley/scratch/DR/classification_performance_files/model7mixed2_correctlyclassified/'
class_num = 0
save_dir = file_dir + str(class_num) + '_saliencymaps/'


def load_images(file_directory, class_name):
    directory = file_directory + str(class_name) + '/'
    img_list = os.listdir(directory)
    
    images_list = []
    for img in img_list:
        images_list.append(np.array(load_img(directory+img, color_mode = 'rgb', target_size=(256,256))))
    images = np.asarray(images_list)

    
    return images, img_list


def model_modifier_function(cloned_model):
    cloned_model.layers[-1].activation = tf.keras.activations.linear
    
# def save_images(images, attention_maps):

#     for i in range(images.shape[0]):
#         fig, axes = plt.subplots(1, 2)
#         axes[0].imshow(images[i], cmap='gray')
#         axes[0].set_title('Original Image')
#         axes[0].axis('off')
#         axes[1].imshow(attention_maps[i], cmap='jet')
#         axes[1].set_title('Saliency Map/Activation')
#         axes[1].axis('off')
#         plt.show()
    
#     return

#%%

model_dir = '/Users/emmastanley/Documents/BME/Research/DR/TCAV/models/13_inception_full_weights.h5'
# model_dir = '/home/estanley/scratch/DR/models/7_inception_mixed2_resume_weights'
model = tf.keras.models.load_model(model_dir)
#%%
# Create Saliency object.
saliency = Saliency(model,
                    model_modifier=model_modifier_function,
                    clone=True)
gradcam = Gradcam(model, model_modifier=model_modifier_function, clone = False)

# save_dir = '/Users/emmastanley/Documents/BME/Research/DR/saliency/model11mixed3/'
# save_dir ='/home/estanley/scratch/DR/saliency/model7mixed2/'


images, image_list = load_images(file_dir, class_num)
def score_function(output): # output shape is (batch_size, num_of_classes)
   return output[:, class_num]


images = images.astype('float32')


for i, ID in enumerate(image_list):
    # saliency_map = saliency(score_function,
    #                     images[i,:,:,:],
    #                     smooth_samples=20, # The number of calculating gradients iterations.
    #                     smooth_noise=0.20) # noise spread level.
    saliency_map = gradcam(score_function,
                        images[i,:,:,:],
                        penultimate_layer=-1)
    
    

    #overlay smoothgrad with image
    heatmap = cm.jet(saliency_map[0,:,:])
    plt.imshow(images[i,:,:,:]/255, cmap='gray')
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.title(image_list[i])
    plt.show()
    # plt.savefig(save_dir + 'saliency_' + image_list[i], dpi=200, bbox_inches='tight')

    # show original image as well
    # plt.imshow(images[i,:,:,:]/255)
    # plt.axis('off')
    # plt.title(image_list[i])
    # plt.show()

    
    