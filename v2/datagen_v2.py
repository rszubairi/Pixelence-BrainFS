# -*- coding: utf-8 -*-
"""
3D Multi-modal Data Generator for Fat Suppression v2

This module provides data generators for 3D U-Net training with support for:
- 3D volumetric data processing
- Multi-modal MRI sequence fusion
- Enhanced data augmentation for 3D volumes
"""

import keras
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from scipy.ndimage import zoom
import skimage
import tensorflow as tf


class DataGenerator3D(keras.utils.Sequence):
    """
    3D Data Generator for volumetric MRI data
    """
    def __init__(self, paths, batch_size=2, input_dims=(32, 128, 128, 3),
                 output_dims=(32, 128, 128, 1), shuffle=True, num_modalities=1,
                 volume_depth=32):
        """
        Args:
            paths: Path to data directory
            batch_size: Batch size for training
            input_dims: Input dimensions (depth, height, width, channels)
            output_dims: Output dimensions (depth, height, width, channels)
            shuffle: Whether to shuffle data
            num_modalities: Number of input modalities
            volume_depth: Depth of 3D volumes
        """
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.batch_size = batch_size
        self.num_modalities = num_modalities
        self.volume_depth = volume_depth
        self.list_IDs = self.get_volume_sequences(paths)
        print('Total no. found volumes: {}.'.format(len(self.list_IDs[0])))
        print('Max allowed iterations: {} '.format(self.__len__()))
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs[0]) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp_fat = [self.list_IDs[0][k] for k in indexes]
        list_IDs_temp_suppressed = [self.list_IDs[1][k] for k in indexes]

        X, y = self.__data_generation(list_IDs_temp_fat, list_IDs_temp_suppressed)
        return X, y

    def get_volume_sequences(self, path):
        """Get volume sequences from directory structure"""
        fat_volumes, suppressed_volumes = [], []

        # Look for volume directories or files
        fat_pattern = os.path.join(path, 'fat', '*')
        suppressed_pattern = os.path.join(path, 'suppressed', '*')

        fat_files = sorted(glob.glob(fat_pattern))
        suppressed_files = sorted(glob.glob(suppressed_pattern))

        # Group files into volumes (assuming sequential naming)
        fat_volumes = self._group_into_volumes(fat_files, 'fat')
        suppressed_volumes = self._group_into_volumes(suppressed_files, 'suppressed')

        min_volumes = min(len(fat_volumes), len(suppressed_volumes))
        return fat_volumes[:min_volumes], suppressed_volumes[:min_volumes]

    def _group_into_volumes(self, files, modality):
        """Group individual slices into 3D volumes"""
        volumes = []
        current_volume = []
        current_base = None

        for file in files:
            filename = os.path.basename(file)
            # Extract volume identifier (assuming naming convention)
            base_name = filename.split('_')[0] if '_' in filename else filename.split('.')[0]

            if current_base != base_name and current_volume:
                volumes.append(current_volume)
                current_volume = []

            current_base = base_name
            current_volume.append(file)

            # If we have enough slices for a volume, create it
            if len(current_volume) >= self.volume_depth:
                volumes.append(current_volume[:self.volume_depth])
                current_volume = []

        # Add remaining slices if they form a partial volume
        if current_volume:
            volumes.append(current_volume)

        return volumes

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs[0]))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def normalisation(self, image):
        """Normalize image to [0,1] range"""
        image = image.astype('float32')
        min_val = np.min(image)
        max_val = np.max(image)
        epsilon = 1e-7
        image = (image - min_val) / (max_val - min_val + epsilon)
        return image

    def rescale_3d(self, volume, target_shape):
        """Rescale 3D volume to target shape"""
        if len(volume.shape) == 4:  # (depth, height, width, channels)
            new_volume = zoom(volume, (
                target_shape[0] / volume.shape[0],
                target_shape[1] / volume.shape[1],
                target_shape[2] / volume.shape[2],
                1
            ))
        else:  # 2D + channels
            new_volume = zoom(volume, (
                target_shape[0] / volume.shape[0],
                target_shape[1] / volume.shape[1],
                target_shape[2] / volume.shape[2]
            ))
        return new_volume

    def load_volume(self, file_list):
        """Load a 3D volume from list of 2D slice files"""
        slices = []
        for file in file_list:
            try:
                slice_data = skimage.io.imread(file, as_gray=True)
                slices.append(slice_data)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue

        if not slices:
            raise ValueError("No valid slices found in volume")

        volume = np.stack(slices, axis=0)
        return volume

    def __data_generation(self, list_IDs_temp_fat, list_IDs_temp_suppressed):
        """Generate batch of 3D data"""
        X = np.empty((self.batch_size, *self.input_dims))
        y = np.empty((self.batch_size, *self.output_dims))

        for n_, (fat_files, suppressed_files) in enumerate(zip(list_IDs_temp_fat, list_IDs_temp_suppressed)):
            try:
                # Load 3D volumes
                fat_volume = self.load_volume(fat_files)
                suppressed_volume = self.load_volume(suppressed_files)

                # Ensure minimum depth
                min_depth = min(fat_volume.shape[0], suppressed_volume.shape[0], self.volume_depth)
                fat_volume = fat_volume[:min_depth]
                suppressed_volume = suppressed_volume[:min_depth]

                # Rescale to target dimensions
                fat_volume = self.rescale_3d(fat_volume, self.input_dims[:3])
                suppressed_volume = self.rescale_3d(suppressed_volume, self.output_dims[:3])

                # Normalize
                fat_volume = self.normalisation(fat_volume)
                suppressed_volume = self.normalisation(suppressed_volume)

                # Expand dimensions for channels if needed
                if len(fat_volume.shape) == 3:
                    fat_volume = np.expand_dims(fat_volume, axis=-1)
                if len(suppressed_volume.shape) == 3:
                    suppressed_volume = np.expand_dims(suppressed_volume, axis=-1)

                # Replicate channels for RGB input if single channel
                if self.input_dims[-1] > fat_volume.shape[-1]:
                    fat_volume = np.repeat(fat_volume, self.input_dims[-1] // fat_volume.shape[-1], axis=-1)

                # Handle multi-modal input
                if self.num_modalities > 1:
                    # For multi-modal, we would load additional modalities here
                    # For now, replicate the same modality
                    modal_volumes = []
                    for _ in range(self.num_modalities):
                        modal_volumes.append(fat_volume[..., :self.input_dims[-1]//self.num_modalities])
                    X[n_] = np.concatenate(modal_volumes, axis=-1)
                else:
                    X[n_] = fat_volume

                y[n_] = suppressed_volume

            except Exception as e:
                print(f"Error processing volume {n_}: {e}")
                # Fill with zeros on error
                X[n_] = np.zeros(self.input_dims)
                y[n_] = np.zeros(self.output_dims)

        return X, y


class MultiModalDataGenerator(keras.utils.Sequence):
    """
    Multi-modal data generator supporting multiple MRI sequences
    """

    def __init__(self, paths, batch_size=2, input_dims=(32, 128, 128, 6),
                 output_dims=(32, 128, 128, 1), shuffle=True,
                 modalities=['t1', 't2'], volume_depth=32):
        """
        Args:
            paths: Dictionary of paths for each modality
            modalities: List of modality names
            Other args same as DataGenerator3D
        """
        self.modalities = modalities
        self.num_modalities = len(modalities)
        self.volume_depth = volume_depth

        # Create generators for each modality
        self.generators = {}
        for modality in modalities:
            if modality in paths:
                gen_dims = (input_dims[0], input_dims[1], input_dims[2],
                           input_dims[3] // self.num_modalities)
                self.generators[modality] = DataGenerator3D(
                    paths[modality], batch_size, gen_dims, output_dims,
                    shuffle, 1, volume_depth
                )

        self.batch_size = batch_size
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.shuffle = shuffle

        # Use the first generator's length as reference
        first_gen = list(self.generators.values())[0]
        self.__len__ = first_gen.__len__

    def __len__(self):
        return self.__len__()

    def __getitem__(self, index):
        # Get batches from all modalities
        X_modalities = []
        y = None

        for modality in self.modalities:
            if modality in self.generators:
                X_mod, y_mod = self.generators[modality][index]
                X_modalities.append(X_mod)
                if y is None:
                    y = y_mod

        # Concatenate modalities along channel dimension
        X = np.concatenate(X_modalities, axis=-1)
        return X, y

    def on_epoch_end(self):
        for generator in self.generators.values():
            generator.on_epoch_end()


# Utility functions for data visualization and validation
def visualize_3d_volume(volume, title="3D Volume", slices_to_show=8):
    """Visualize slices from a 3D volume"""
    depth = volume.shape[0]
    slice_indices = np.linspace(0, depth-1, slices_to_show, dtype=int)

    fig, axes = plt.subplots(1, slices_to_show, figsize=(20, 4))
    fig.suptitle(title)

    for i, slice_idx in enumerate(slice_indices):
        if len(volume.shape) == 4:
            axes[i].imshow(volume[slice_idx, ..., 0], cmap='gray')
        else:
            axes[i].imshow(volume[slice_idx], cmap='gray')
        axes[i].set_title(f'Slice {slice_idx}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def sanity_check_3d_generator(generator, num_samples=2):
    """Perform sanity check on 3D data generator"""
    print("Performing 3D generator sanity check...")

    for i in range(min(num_samples, len(generator))):
        try:
            X, y = generator[i]
            print(f"Batch {i}: X shape: {X.shape}, y shape: {y.shape}")
            print(f"X range: [{X.min():.3f}, {X.max():.3f}]")
            print(f"y range: [{y.min():.3f}, {y.max():.3f}]")

            # Visualize first volume in batch
            if X.shape[0] > 0:
                visualize_3d_volume(X[0], f"Input Volume {i}")
                visualize_3d_volume(y[0], f"Target Volume {i}")

        except Exception as e:
            print(f"Error in batch {i}: {e}")

    print("Sanity check completed.")
