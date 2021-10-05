#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 14:37:01 2021

@author: emmastanley
"""
import tcav.activation_generator as act_gen
import tcav.cav as cav
import tcav.model  as model
import tcav.tcav as tcav
import tcav.utils as utils
#import tcav.utils_plot as utils_plot # utils_plot requires matplotlib
import os 
import tensorflow as tf
import tcav.utils_saveplot as utils_saveplot

#path containing subdirectories for random images, concepts, class examples, and labels
#Deepthi - this is whereever you save all the contents of the OneDrive folder I sent you 
source_dir='/home/estanley/scratch/TCAV/TCAV_cropped_images'

# the name of the parent directory that results (CAVs and activations) are stored (only if you want to cache)
project_name = '/tcav_test'
working_dir=source_dir +project_name

#folder for activations to be stored - if your act_gen_wrapper does so 
activation_dir=working_dir+ '/activations/'
#folder for cavs to be stored, can say "None" if you don't want to store them
cav_dir = working_dir + '/cavs/'


#list of names of bottleneck layers that you want to use for TCAV - defined in the model wrapper
bottlenecks = ['mixed0', 'mixed1', 'mixed2', 'mixed3', 'mixed4', 'mixed5', 'mixed6', 'mixed7', 'mixed8', 'mixed9', 'mixed10'] #you can add more layers to the list here but my computer isnt able to do that much processing, gives me a thread count error

#make the directories
utils.make_dir_if_not_exists(activation_dir)
utils.make_dir_if_not_exists(working_dir)
utils.make_dir_if_not_exists(cav_dir)

# this is a regularizer penalty parameter for linear classifier to get CAVs. 
alphas = [0.1]   

#target and concepts are the folder names in source_dir
target = 'class2' #class we are investigating
concepts = ["hemmorhage_cropped", "microaneursym_cropped", "hardexudate_cropped", "tortuous_cropped", "softexudate_cropped"] #concepts we are investigating

#%%

      
#Create Tensorflow session
sess = utils.create_session()

#Deepthi - this model is based on the InceptionV3 archicture, I just added an additional dense layer and did transfer learning with it
mymodel = tf.keras.models.load_model(source_dir + "/models/4_inception_mixed4_weights.h5") 

'''
the following is a model wrapper that needs to be defined for use with the TCAV module.
in the TCAV github repo under TCAV/model.py there are frameworks for writing custom wrappers but i didn't know how to do that lol 
so I found this one online for InceptionV3 and it appears to work, i've been going through it trying to understand
'''
# Modified version of PublicImageModelWrapper in TCAV's models.py
# This class takes a session which contains the already loaded graph.
# This model also assumes softmax is used with categorical crossentropy.
#https://gist.github.com/Gareth001/e600d2fbc09e690c4333388ec5f06587
class CustomPublicImageModelWrapper(model.ImageModelWrapper):
    def __init__(self, sess, labels, image_shape,
                endpoints_dict, name, image_value_range):
        super(self.__class__, self).__init__(image_shape)
        
        self.sess = sess
        self.labels = labels
        self.model_name = name
        self.image_value_range = image_value_range

        # get endpoint tensors
        self.ends = {'input': endpoints_dict['input_tensor'], 'prediction': endpoints_dict['prediction_tensor']}
        
        self.bottlenecks_tensors = self.get_bottleneck_tensors()
        
        # load the graph from the backend
        # graph = tf.get_default_graph()
        graph = tf.compat.v1.get_default_graph() #tf 2.x

        # Construct gradient ops.
        with graph.as_default():
            # self.y_input = tf.placeholder(tf.int64, shape=[None])
            self.y_input = tf.compat.v1.placeholder(tf.int64, shape=[None]) #tf 2.x

            self.pred = tf.expand_dims(self.ends['prediction'][0], 0)
            self.loss = tf.reduce_mean(
                # tf.nn.softmax_cross_entropy_with_logits_v2(
                tf.nn.softmax_cross_entropy_with_logits( #tf 2.x
                    labels=tf.one_hot(
                        self.y_input,
                        self.ends['prediction'].get_shape().as_list()[1]),
                    logits=self.pred))
        self._make_gradient_tensors()

    def id_to_label(self, idx):
        return self.labels[idx]

    def label_to_id(self, label):
        return self.labels.index(label)

    @staticmethod
    def create_input(t_input, image_value_range):
        """Create input tensor."""
        def forget_xy(t):
            """Forget sizes of dimensions [1, 2] of a 4d tensor."""
            zero = tf.identity(0)
            return t[:, zero:, zero:, :]

        t_prep_input = t_input
        if len(t_prep_input.shape) == 3:
            t_prep_input = tf.expand_dims(t_prep_input, 0)
        t_prep_input = forget_xy(t_prep_input)
        lo, hi = image_value_range
        t_prep_input = lo + t_prep_input * (hi-lo)
        return t_input, t_prep_input

    @staticmethod
    def get_bottleneck_tensors():
        """Add Inception bottlenecks and their pre-Relu versions to endpoints dict."""
        # graph = tf.get_default_graph()
        graph = tf.compat.v1.get_default_graph() #tf 2.x
        bn_endpoints = {}
        for op in graph.get_operations():
            # change this below string to change which layers are considered bottlenecks
            # use 'ConcatV2' for InceptionV3
            # use 'MaxPool' for VGG16 (for example)
            if 'ConcatV2' in op.type: 
                name = op.name.split('/')[0]
                bn_endpoints[name] = op.outputs[0]
            
        return bn_endpoints
      
endpoints_v3 = dict(
    input=mymodel.inputs[0].name,
    input_tensor=mymodel.inputs[0],
    logit=mymodel.outputs[0].name,
    prediction=mymodel.outputs[0].name,
    prediction_tensor=mymodel.outputs[0],
)



mymodel_wrapped = CustomPublicImageModelWrapper(sess, 
        ['class0', 'class1', 'class2', 'class3', 'class4'], [256, 256, 3], endpoints_v3, 
        'InceptionV3_public', (-1, 1))



#implement a class of ActivationGenerationInterface which TCAV uses to load example data for a given concept/target, call into model wrapper, and return activations
act_generator = act_gen.ImageActivationGenerator(mymodel_wrapped, source_dir, activation_dir, max_examples=100)

#%%

tf.compat.v1.logging.set_verbosity(0)

#the number of experiments run to confirm meaningful concept direction 
num_random_exp=20 #just 2 for now to keep it fast, but should be at least 10-20 times for meaningful results. the paper used 500

mytcav = tcav.TCAV(sess,
                   target,
                   concepts,
                   bottlenecks,
                   act_generator,
                   alphas,
                   cav_dir=cav_dir,
                   num_random_exp=num_random_exp)

print ('This may take a while... Go get coffee!')

results = mytcav.run(run_parallel=False)


print ('done!')

savedir = '/home/estanley/scratch/TCAV/plots/'
plotname = 'tcav_results_test02'

utils_saveplot.plot_results(results, savedir = savedir, plotname = plotname , num_random_exp=num_random_exp) #plot the TCAV scores and display results










