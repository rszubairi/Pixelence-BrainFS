# -*- coding: utf-8 -*-
"""
Utilities for 3D Fat Suppression v2

This module provides utility functions adapted for 3D volumes:
- Loss functions for 3D data
- Metrics for 3D volumes
- Visualization functions
- Performance callbacks
"""

import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras.applications.vgg19 import VGG19
import keras.backend as K


def calculate_ssim_3d(y_true, y_pred, max_val=1.0):
    """
    Calculate SSIM for 3D volumes
    """
    # Ensure both tensors are the same dtype
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    max_val = tf.cast(max_val, tf.float32)
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=max_val, filter_size=5))


def calculate_psnr_3d(y_true, y_pred, max_val=1.0):
    """
    Calculate PSNR for 3D volumes
    """
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=max_val))


# Load VGG for 2D perceptual loss (will be adapted for 3D)
VGG = VGG19(input_shape=None, weights='imagenet', include_top=False)
VGG.trainable = False

# Upsampler for 2D projections
upsampler_2d = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')


@tf.function
def VGG_loss_3d(y_true, y_pred):
    """
    3D VGG perceptual loss using 2D projections
    """
    # Extract 2D slices from 3D volumes for perceptual loss
    # Use middle slices in each spatial dimension

    def extract_2d_slices(volume):
        """Extract representative 2D slices from 3D volume"""
        # volume shape: (batch, depth, height, width, channels)

        # Axial view (middle slice in depth dimension)
        axial = volume[:, volume.shape[1]//2, :, :, :]  # Shape: (batch, height, width, channels)

        # Sagittal view (middle slice in width dimension)
        sagittal = volume[:, :, volume.shape[2]//2, :, :]  # Shape: (batch, depth, width, channels)

        # Coronal view (middle slice in height dimension)
        coronal = volume[:, :, :, volume.shape[3]//2, :]  # Shape: (batch, depth, height, channels)

        return axial, sagittal, coronal

    # Extract slices from both true and predicted volumes
    true_axial, true_sagittal, true_coronal = extract_2d_slices(y_true)
    pred_axial, pred_sagittal, pred_coronal = extract_2d_slices(y_pred)

    # Calculate VGG loss for each projection
    def vgg_loss_2d(true_2d, pred_2d):
        # Ensure 3 channels for VGG input
        if true_2d.shape[-1] == 1:
            true_2d = tf.repeat(true_2d, 3, axis=-1)
            pred_2d = tf.repeat(pred_2d, 3, axis=-1)

        # Apply VGG
        true_features = VGG(true_2d)
        pred_features = VGG(pred_2d)

        # Calculate feature-level loss
        h1 = tf.reshape(true_features, (tf.shape(true_features)[0], -1))
        h2 = tf.reshape(pred_features, (tf.shape(pred_features)[0], -1))
        return tf.reduce_sum(tf.square(h1 - h2), axis=-1)

    # Combine losses from all projections
    axial_loss = vgg_loss_2d(true_axial, pred_axial)
    sagittal_loss = vgg_loss_2d(true_sagittal, pred_sagittal)
    coronal_loss = vgg_loss_2d(true_coronal, pred_coronal)

    # Average across projections and batch
    total_loss = (axial_loss + sagittal_loss + coronal_loss) / 3.0
    return tf.reduce_mean(total_loss)


@tf.function
def combined_loss_3d(y_true, y_pred, lambda_perceptual=1.0, lambda_l1=0.1):
    """
    Combined loss for 3D volumes: perceptual + L1
    """
    perceptual_loss = VGG_loss_3d(y_true, y_pred)
    l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))

    return lambda_perceptual * perceptual_loss + lambda_l1 * l1_loss


@tf.function
def dice_loss_3d(y_true, y_pred, smooth=1e-6):
    """
    Dice loss for 3D volumes (useful for segmentation-like tasks)
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


class PerformanceCallback3D(keras.callbacks.Callback):
    """
    Performance callback for 3D volumes with multi-planar visualization
    """
    def __init__(self, data_gen, save_path='training_visualizations'):
        self.data_gen = data_gen
        self.epoch = 0
        self.save_path = save_path
        import os
        os.makedirs(save_path, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1
        try:
            # Get a batch for visualization
            fat_vol, suppresed_vol = self.data_gen[0]

            # Generate prediction
            gen_vol = self.model.predict(fat_vol, verbose=0)

            # Visualize middle slices
            self.visualize_3d_results(fat_vol[0], suppresed_vol[0], gen_vol[0], epoch)

        except Exception as e:
            print(f"Error in performance callback: {e}")

    def visualize_3d_results(self, input_vol, target_vol, pred_vol, epoch):
        """Visualize 3D results with multi-planar reconstruction"""
        # Take middle slices in each dimension
        depth_mid = input_vol.shape[0] // 2
        height_mid = input_vol.shape[1] // 2
        width_mid = input_vol.shape[2] // 2

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(f'3D Results - Epoch {epoch}', fontsize=16)

        # Axial view (middle slice in depth)
        axes[0,0].imshow(input_vol[depth_mid, :, :, 0], cmap='gray')
        axes[0,0].set_title('Input (Axial)')
        axes[0,0].axis('off')

        axes[0,1].imshow(target_vol[depth_mid, :, :, 0], cmap='gray')
        axes[0,1].set_title('Target (Axial)')
        axes[0,1].axis('off')

        axes[0,2].imshow(pred_vol[depth_mid, :, :, 0], cmap='gray')
        axes[0,2].set_title('Prediction (Axial)')
        axes[0,2].axis('off')

        # Sagittal view (middle slice in width)
        axes[1,0].imshow(input_vol[:, height_mid, :, 0], cmap='gray')
        axes[1,0].set_title('Input (Sagittal)')
        axes[1,0].axis('off')

        axes[1,1].imshow(target_vol[:, height_mid, :, 0], cmap='gray')
        axes[1,1].set_title('Target (Sagittal)')
        axes[1,1].axis('off')

        axes[1,2].imshow(pred_vol[:, height_mid, :, 0], cmap='gray')
        axes[1,2].set_title('Prediction (Sagittal)')
        axes[1,2].axis('off')

        # Coronal view (middle slice in height)
        axes[2,0].imshow(input_vol[:, :, width_mid, 0], cmap='gray')
        axes[2,0].set_title('Input (Coronal)')
        axes[2,0].axis('off')

        axes[2,1].imshow(target_vol[:, :, width_mid, 0], cmap='gray')
        axes[2,1].set_title('Target (Coronal)')
        axes[2,1].axis('off')

        axes[2,2].imshow(pred_vol[:, :, width_mid, 0], cmap='gray')
        axes[2,2].set_title('Prediction (Coronal)')
        axes[2,2].axis('off')

        plt.tight_layout()

        # Save the figure
        save_file = f"{self.save_path}/epoch_{epoch:03d}.png"
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved visualization: {save_file}")


def plot_training_history_3d(history, save_path='training_history_v2.png'):
    """Plot training history with 3D-specific metrics"""
    metrics = ['loss', 'mae', 'calculate_ssim_3d', 'calculate_psnr_3d']
    names_metrics = ['Combined Loss', 'MAE', 'SSIM', 'PSNR']
    epochs = range(1, len(history.history['loss']) + 1)

    plt.figure(figsize=(16, 12))

    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)
        plt.plot(epochs, history.history[metric], 'b-', label=f'Training {names_metrics[i]}', linewidth=2)
        val_metric = f'val_{metric}'
        if val_metric in history.history:
            plt.plot(epochs, history.history[val_metric], 'r-', label=f'Validation {names_metrics[i]}', linewidth=2)
        plt.title(f'{names_metrics[i]} Over Training', fontsize=14, fontweight='bold')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel(names_metrics[i], fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training history plot saved to {save_path}")


def create_data_augmentation_3d():
    """
    Create data augmentation pipeline for 3D volumes
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip(mode="horizontal"),
        tf.keras.layers.RandomRotation(factor=0.1, fill_mode='nearest'),
        tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1, fill_mode='nearest'),
        # Add noise
        tf.keras.layers.GaussianNoise(stddev=0.01),
    ])


def volume_statistics(volume):
    """
    Calculate statistics for a 3D volume
    """
    stats = {
        'mean': float(np.mean(volume)),
        'std': float(np.std(volume)),
        'min': float(np.min(volume)),
        'max': float(np.max(volume)),
        'shape': volume.shape,
        'dtype': str(volume.dtype)
    }
    return stats


def validate_3d_data_generator(generator, num_samples=5):
    """
    Validate 3D data generator by checking data properties
    """
    print("Validating 3D data generator...")

    stats_input = []
    stats_target = []

    for i in range(min(num_samples, len(generator))):
        try:
            X, y = generator[i]

            # Calculate statistics
            input_stats = volume_statistics(X[0])  # First volume in batch
            target_stats = volume_statistics(y[0])

            stats_input.append(input_stats)
            stats_target.append(target_stats)

            print(f"Batch {i}: X shape {X.shape}, y shape {y.shape}")
            print(f"  Input - Mean: {input_stats['mean']:.3f}, Std: {input_stats['std']:.3f}")
            print(f"  Target - Mean: {target_stats['mean']:.3f}, Std: {target_stats['std']:.3f}")

        except Exception as e:
            print(f"Error in batch {i}: {e}")

    # Summary statistics
    if stats_input:
        print("\nInput Statistics Summary:")
        print(f"Mean range: [{min(s['mean'] for s in stats_input):.3f}, {max(s['mean'] for s in stats_input):.3f}]")
        print(f"Std range: [{min(s['std'] for s in stats_input):.3f}, {max(s['std'] for s in stats_input):.3f}]")

    if stats_target:
        print("\nTarget Statistics Summary:")
        print(f"Mean range: [{min(s['mean'] for s in stats_target):.3f}, {max(s['mean'] for s in stats_target):.3f}]")
        print(f"Std range: [{min(s['std'] for s in stats_target):.3f}, {max(s['std'] for s in stats_target):.3f}]")

    print("Validation completed.")


# Legacy compatibility - import original functions
try:
    from utils import PerformanceCallback_Plot, calculate_ssim, calculate_psnr
except ImportError:
    # Fallback if utils.py is not available
    PerformanceCallback_Plot = None
    calculate_ssim = None
    calculate_psnr = None

VGG_loss = VGG_loss_3d  # Override with 3D version
