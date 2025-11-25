# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 21:39:11 2024

@author: amraa
"""
from models import build_generator
from datagen import DataGenerator
from utils import PerformanceCallback_Plot, calculate_ssim, calculate_psnr, VGG_loss
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

# config
fat_dims = (256, 256, 3)
suppressed_dims = (256, 256, 1)

# data generators
path_train = 'Fat Suppresion/train/'
data_gen_train = DataGenerator(paths=path_train,
                         batch_size=6,
                         input_dims=fat_dims,
                         output_dims = suppressed_dims,
                         shuffle=True)

path_valid= 'Fat Suppresion/valid/'
data_gen_valid = DataGenerator(paths=path_valid,
                         batch_size=6,
                         input_dims=fat_dims,
                         output_dims = suppressed_dims,
                         shuffle=True)


# model
model = build_generator(fat_dims, 22)
#model.load_weights('F:/Python projects/Image_Processing/05 - Image Conversion/Convert MRI T1 non-cont to T1+Con/Model V1 - On publication/weights - Best T13.h5')


optimizer = tf.keras.optimizers.Adamax(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss = VGG_loss, metrics=['mae', calculate_ssim, calculate_psnr])
model.summary()

# callbacks
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.99,
                                                 patience=10,
                                                 min_lr=1e-8)

earlystoping = tf.keras.callbacks.EarlyStopping('val_loss',
                                                patience=25)

modelCheckpoint = tf.keras.callbacks.ModelCheckpoint('best_weights.weights.h5',
                                           monitor="val_loss",
                                           verbose=0,
                                           save_best_only=True,
                                           save_weights_only=True)

performancecallback_plot = PerformanceCallback_Plot(data_gen_valid)

# training
history = model.fit(data_gen_train, validation_data=data_gen_valid, epochs=2000, callbacks=[reduce_lr, earlystoping, performancecallback_plot, modelCheckpoint])

pd.DataFrame(history.history).to_csv('model_history.csv')

# performance
def plot_metrics(history):
    metrics = ['loss', 'mae', 'calculate_ssim', 'calculate_psnr']
    names_metrics = ['VGG loss', 'MAE', 'Structural Similarity Index Measure', 'Peak Signal-to-Noise Ratio']
    epochs = range(1, len(history.history['loss']) + 1)
    
    plt.figure(figsize=(14, 10))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)
        plt.plot(epochs, history.history[metric], 'b', label=f'Training {names_metrics[i]}')
        plt.plot(epochs, history.history[f'val_{metric}'], 'r', label=f'Validation {names_metrics[i]}')
        plt.title(f'Training and Validation {names_metrics[i].upper()}')
        plt.xlabel('Epochs')
        plt.ylabel(metric.upper())
        plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_metrics(history)