# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 21:42:58 2024

@author: amraa
"""

import keras
import matplotlib.pyplot as plt
import numpy as np
import glob
import keras
from scipy.ndimage import zoom
import skimage

dataset_ar_path = 'I:\Datasets\Fat Suppresion/valid/'

## data preprocessing and generator class
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, paths, batch_size = 4, input_dims=(128, 128, 3), output_dims =(128, 128, 3), shuffle=True):
        'Initialization'
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.batch_size = batch_size 
        self.list_IDs = self.get_squences_ids(paths)
        print('total no. found inside the path dirs is: {} Cases.'.format(len(self.list_IDs[0])))
        print('Max allowed iterations: {} '.format(self.__len__()))
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs[0]) / self.batch_size))

    def __call__(self):
        self.__getitem__(np.random.randint(0, self.__len__()))
        
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp_fat = [self.list_IDs[0][k] for k in indexes]
        list_IDs_temp_suppresed = [self.list_IDs[1][k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp_fat, list_IDs_temp_suppresed)

        return X, y

    def get_squences_ids(self, path):      
        fat, suppressed = [], []
        for fold, ls in zip(['/fat/*', '/suppressed/*'], [fat, suppressed]):
            ls.extend(glob.glob(path + fold))
        return fat, suppressed
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs[0]))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def normalisation(self, image):
        image = image.astype('float32')
        min_val = np.min(image)
        max_val = np.max(image)
        
        # Normalize the pixel values
        epsilon = 1e-7
        image = (image - min_val) / (max_val - min_val + epsilon)
        return image
        
    def rescale(self, arr, dim=128):
        new_ar = zoom(arr, (dim/arr.shape[0], dim/arr.shape[1], 1))
        return new_ar 
    
    def read_jpg(self, img):
        return np.expand_dims(skimage.io.imread(img, as_gray=True), -1)
    
    def __data_generation(self, list_IDs_temp_fat, list_IDs_temp_suppressed):
        'Generates data containing batch_size samples'
        # Initialization
        
        X = np.empty((self.batch_size, *self.input_dims))
        y = np.empty((self.batch_size, *self.output_dims))
        
        # Generate data
        for n_, (fat_, suppressed_) in enumerate(zip(list_IDs_temp_fat, list_IDs_temp_suppressed)):
            X[n_] = np.repeat(self.normalisation(self.rescale(self.read_jpg(fat_), self.input_dims[0])), 3, -1)
            y[n_] = self.normalisation(self.rescale(self.read_jpg(suppressed_), self.input_dims[0]))
        return X, y

# sanity check
xr = 256
input_shape = (xr, xr, 3) 
output_shape = (xr, xr, 1)

train_generator = DataGenerator(dataset_ar_path, batch_size=4, input_dims=input_shape, output_dims= output_shape, shuffle=True)

train_generator.__getitem__(300)
train_generator.__len__()

input_images, output_images = train_generator.__getitem__(1)

   

def gen_sanity_check(gen):
    i = np.random.randint(0, len(gen))
    input_images, output_images = gen.__getitem__(i)
    fig, axes = plt.subplots(1, 2, figsize=(20, 14))
    axes[0].imshow(input_images[2], cmap='gray')
    axes[0].axis('off') 
    axes[0].set_title('Fat') 
    axes[1].imshow(output_images[2].squeeze(), cmap='gray')
    axes[1].axis('off')
    axes[1].set_title('Suppressed')                 
    plt.show()   
                   
gen_sanity_check(train_generator)
