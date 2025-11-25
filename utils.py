# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 02:28:31 2024

@author: amraa
"""

import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras.applications.vgg19 import VGG19
import keras.backend as K

class PerformanceCallback_Plot(keras.callbacks.Callback):
    def __init__(self, data_gen):
        self.data_gen = data_gen
        self.epoch = 0
        
    def __call__(self):
        print('The Performance Callback Plot Class is already Initiated')
                
    def on_epoch_end(self, epoch, logs=None):
        int_ = np.random.randint(0, len(self.data_gen))
        self.epoch += 1
        fat, suppresed = self.data_gen.__getitem__(int_)
        
        gen_img = self.model.predict(fat, verbose = 0)
        Img_titles = ['Fat', 'suppresed truth', 'generated image'] 
        image_list = [fat[0], suppresed[0], gen_img[0]]
        
        fig, axes = plt.subplots(1, 3, facecolor='lightsteelblue', figsize = (10,5))
        fig.suptitle(f'Epoch no: {epoch}\n',
                     fontsize = 18,
                     color = 'darkgoldenrod')
        axes = axes.ravel()
        for i, ax in enumerate((axes)):
            ax.imshow(image_list[i], cmap='gray')
            ax.axis('off')
            ax.set_title(Img_titles[i], fontsize = 18, color='darkblue')
        plt.tight_layout(pad=1.3)
        plt.show()
        
        

def calculate_ssim(image1, image2):
   return tf.reduce_mean(tf.image.ssim(image1, image2, max_val=1.0, filter_size=5)) * 100

def calculate_psnr(image1, image2):
    return tf.reduce_mean(tf.image.psnr(image1, image2, max_val=1.0))

        
VGG=VGG19(input_shape=None,weights='imagenet', include_top=False)
VGG.trainable = False

upsampler = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')

@tf.function
def VGG_loss(y_true, y_pred):
    y_true = VGG(tf.repeat(upsampler(y_true), 3, axis=-1))
    y_pred = VGG(tf.repeat(upsampler(y_pred), 3, axis=-1))
    
    
    h1 = K.batch_flatten(y_true)
    h2 = K.batch_flatten(y_pred)
    rc_loss =  K.sum(K.square(h1 - h2), axis=-1)
    return rc_loss  