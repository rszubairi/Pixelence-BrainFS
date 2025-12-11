# -*- coding: utf-8 -*-
"""
Training Script for 3D Multi-modal Fat Suppression v2

This script implements training for:
- 3D U-Net with spatial consistency
- Multi-modal learning with MRI sequence fusion
- Enhanced attention mechanisms
"""

from models_v2 import build_3d_generator, build_multimodal_discriminator
from datagen_v2 import DataGenerator3D, MultiModalDataGenerator, NiftiBraTSDataGenerator, RemindDICOMDataGenerator, sanity_check_3d_generator
import pydicom
from utils_v2 import PerformanceCallback3D, calculate_ssim_3d, calculate_psnr_3d, VGG_loss_3d as VGG_loss
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import argparse


def create_gan(generator, discriminator):
    """Create GAN model combining generator and discriminator"""
    discriminator.trainable = False
    gan_input = tf.keras.Input(shape=generator.input_shape[1:])
    generated = generator(gan_input)
    gan_output = discriminator(generated)
    gan = tf.keras.Model(gan_input, gan_output)
    return gan


def train_3d_unet(config):
    """Train 3D U-Net model"""
    print("=== Training 3D U-Net for Fat Suppression v2 ===")
    print(f"Input shape: {config['input_shape']}")
    print(f"Output shape: {config['output_shape']}")
    print(f"Modalities: {config['num_modalities']}")
    print(f"Use NIfTI data: {config.get('use_nifti', False)}")
    print(f"Use REMIND DICOM data: {config.get('use_remind', False)}")

    # Create data generators
    if config.get('use_remind', False):
        # Use REMIND DICOM data generator
        # Create a temporary generator to get the split
        temp_gen = RemindDICOMDataGenerator(
            data_path=config['data_path'],
            batch_size=config['batch_size'],
            input_dims=config['input_shape'],
            output_dims=config['output_shape'],
            validation_split=0.2
        )
        # Training generator
        data_gen_train = RemindDICOMDataGenerator(
            data_path=config['data_path'],
            batch_size=config['batch_size'],
            input_dims=config['input_shape'],
            output_dims=config['output_shape'],
            validation_split=0.0
        )
        data_gen_train.train_patients = temp_gen.train_patients
        data_gen_train.patient_folders = temp_gen.train_patients
        data_gen_train.on_epoch_end()

        # Validation generator
        data_gen_valid = RemindDICOMDataGenerator(
            data_path=config['data_path'],
            batch_size=config['batch_size'],
            input_dims=config['input_shape'],
            output_dims=config['output_shape'],
            validation_split=0.0
        )
        data_gen_valid.train_patients = temp_gen.val_patients
        data_gen_valid.patient_folders = temp_gen.val_patients
        data_gen_valid.on_epoch_end()

    elif config.get('use_nifti', False):
        # Use NIfTI BraTS data generator
        # Create a temporary generator to get the split
        temp_gen = NiftiBraTSDataGenerator(
            data_path=config['data_path'],
            batch_size=config['batch_size'],
            input_dims=config['input_shape'],
            output_dims=config['output_shape'],
            validation_split=0.2
        )
        # Training generator
        data_gen_train = NiftiBraTSDataGenerator(
            data_path=config['data_path'],
            batch_size=config['batch_size'],
            input_dims=config['input_shape'],
            output_dims=config['output_shape'],
            validation_split=0.0
        )
        data_gen_train.train_patients = temp_gen.train_patients
        data_gen_train.patient_folders = temp_gen.train_patients
        data_gen_train.on_epoch_end()

        # Validation generator
        data_gen_valid = NiftiBraTSDataGenerator(
            data_path=config['data_path'],
            batch_size=config['batch_size'],
            input_dims=config['input_shape'],
            output_dims=config['output_shape'],
            validation_split=0.0
        )
        data_gen_valid.train_patients = temp_gen.val_patients
        data_gen_valid.patient_folders = temp_gen.val_patients
        data_gen_valid.on_epoch_end()

    elif config['num_modalities'] > 1:
        # Multi-modal training
        paths = {f'modality_{i}': config['data_path'] for i in range(config['num_modalities'])}
        data_gen_train = MultiModalDataGenerator(
            paths=paths,
            batch_size=config['batch_size'],
            input_dims=config['input_shape'],
            output_dims=config['output_shape'],
            modalities=[f'modality_{i}' for i in range(config['num_modalities'])],
            volume_depth=config['volume_depth']
        )
        data_gen_valid = MultiModalDataGenerator(
            paths={f'modality_{i}': config['valid_path'] for i in range(config['num_modalities'])},
            batch_size=config['batch_size'],
            input_dims=config['input_shape'],
            output_dims=config['output_shape'],
            modalities=[f'modality_{i}' for i in range(config['num_modalities'])],
            volume_depth=config['volume_depth']
        )
    else:
        # Single modality 3D training
        data_gen_train = DataGenerator3D(
            paths=config['data_path'],
            batch_size=config['batch_size'],
            input_dims=config['input_shape'],
            output_dims=config['output_shape'],
            num_modalities=config['num_modalities'],
            volume_depth=config['volume_depth']
        )
        data_gen_valid = DataGenerator3D(
            paths=config['valid_path'],
            batch_size=config['batch_size'],
            input_dims=config['input_shape'],
            output_dims=config['output_shape'],
            num_modalities=config['num_modalities'],
            volume_depth=config['volume_depth']
        )

    # Build models
    generator = build_3d_generator(
        config['input_shape'],
        config['filters'],
        num_modalities=config['num_modalities']
    )

    if config['use_gan']:
        discriminator = build_multimodal_discriminator(
            config['output_shape'],
            num_modalities=1  # Discriminator works on generated output
        )
        gan = create_gan(generator, discriminator)

    # Compile models
    optimizer_gen = tf.keras.optimizers.Adamax(learning_rate=config['learning_rate'])

    if config['use_gan']:
        optimizer_disc = tf.keras.optimizers.Adamax(learning_rate=config['learning_rate'])
        discriminator.compile(optimizer=optimizer_disc, loss='binary_crossentropy', metrics=['accuracy'])

        # GAN loss combines reconstruction and adversarial loss
        def gan_loss(y_true, y_pred):
            reconstruction_loss = VGG_loss(y_true, y_pred)
            # Add adversarial loss here if needed
            return reconstruction_loss

        gan.compile(optimizer=optimizer_gen, loss=gan_loss, metrics=['mae', calculate_ssim_3d, calculate_psnr_3d])
        model = gan
    else:
        generator.compile(optimizer=optimizer_gen, loss=VGG_loss, metrics=['mae', calculate_ssim_3d, calculate_psnr_3d])
        model = generator

    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.95,
            patience=10,
            min_lr=1e-8
        ),
        tf.keras.callbacks.EarlyStopping(
            'val_loss',
            patience=25
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_weights_v2.weights.h5',
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=True
        )
    ]

    if not config['use_gan']:
        callbacks.append(PerformanceCallback3D(data_gen_valid))

    # Training
    print(f"Starting training for {config['epochs']} epochs...")
    history = model.fit(
        data_gen_train,
        validation_data=data_gen_valid,
        epochs=config['epochs'],
        callbacks=callbacks
    )

    # Save training history
    pd.DataFrame(history.history).to_csv('model_history_v2.csv')
    print("Training completed. History saved to model_history_v2.csv")

    return history


class PerformanceCallback3D(tf.keras.callbacks.Callback):
    """Performance callback for 3D volumes"""
    def __init__(self, data_gen):
        self.data_gen = data_gen
        self.epoch = 0

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
        """Visualize 3D results"""
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
        plt.show()


def plot_training_history(history, save_path='training_history_v2.png'):
    """Plot training history"""
    metrics = ['loss', 'mae', 'calculate_ssim', 'calculate_psnr']
    names_metrics = ['VGG Loss', 'MAE', 'SSIM', 'PSNR']
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


def main():
    """Main training function with argument parsing"""
    parser = argparse.ArgumentParser(description='Train 3D Multi-modal Fat Suppression v2 - REMIND DICOM Version')
    parser.add_argument('--data_path', type=str, default=r'C:\Users\rszub\Documents\Brain Dataset\Remind\ReMIND',
                       help='Path to REMIND DICOM dataset directory')
    parser.add_argument('--valid_path', type=str, default=r'C:\Users\rszub\Documents\Brain Dataset\Remind\ReMIND',
                       help='Path to REMIND DICOM validation data')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--filters', type=int, default=16,
                       help='Base number of filters in U-Net')
    parser.add_argument('--num_modalities', type=int, default=1,
                       help='Number of input modalities')
    parser.add_argument('--volume_depth', type=int, default=32,
                       help='Depth of 3D volumes')
    parser.add_argument('--volume_height', type=int, default=128,
                       help='Height of 3D volumes')
    parser.add_argument('--volume_width', type=int, default=128,
                       help='Width of 3D volumes')
    parser.add_argument('--use_gan', action='store_true',
                       help='Use GAN training instead of supervised learning')
    parser.add_argument('--use_remind', action='store_true',
                       help='Use REMIND DICOM dataset format (T2 -> FLAIR)')
    parser.add_argument('--sanity_check', action='store_true',
                       help='Run data generator sanity check before training')

    args = parser.parse_args()

    # Create configuration
    config = {
        'data_path': args.data_path,
        'valid_path': args.valid_path,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'filters': args.filters,
        'num_modalities': args.num_modalities,
        'volume_depth': args.volume_depth,
        'input_shape': (args.volume_depth, args.volume_height, args.volume_width,
                       3 * args.num_modalities),  # 3 channels per modality
        'output_shape': (args.volume_depth, args.volume_height, args.volume_width, 1),
        'use_gan': args.use_gan,
        'use_remind': args.use_remind
    }

    # Run sanity check if requested
    if args.sanity_check:
        print("Running data generator sanity check...")
        if config.get('use_remind', False):
            test_gen = RemindDICOMDataGenerator(
                data_path=config['data_path'],
                batch_size=1,
                input_dims=config['input_shape'],
                output_dims=config['output_shape'],
                validation_split=0.2
            )
        elif config.get('use_nifti', False):
            test_gen = NiftiBraTSDataGenerator(
                data_path=config['data_path'],
                batch_size=1,
                input_dims=config['input_shape'],
                output_dims=config['output_shape'],
                validation_split=0.2
            )
        elif config['num_modalities'] > 1:
            paths = {f'modality_{i}': config['data_path'] for i in range(config['num_modalities'])}
            test_gen = MultiModalDataGenerator(
                paths=paths,
                batch_size=1,
                input_dims=config['input_shape'],
                output_dims=config['output_shape'],
                modalities=[f'modality_{i}' for i in range(config['num_modalities'])],
                volume_depth=config['volume_depth']
            )
        else:
            test_gen = DataGenerator3D(
                paths=config['data_path'],
                batch_size=1,
                input_dims=config['input_shape'],
                output_dims=config['output_shape'],
                num_modalities=config['num_modalities'],
                volume_depth=config['volume_depth']
            )

        sanity_check_3d_generator(test_gen, num_samples=2)

    # Train the model
    history = train_3d_unet(config)

    # Plot training history
    plot_training_history(history)


if __name__ == '__main__':
    main()
