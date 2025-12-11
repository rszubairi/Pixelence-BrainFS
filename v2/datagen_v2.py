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
import nibabel as nib


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


class T2WeightedDICOMDataGenerator(keras.utils.Sequence):
    """
    Data Generator for T2-weighted DICOM volumes - T2 as input, FLAIR (T2-weighted) as target
    """

    def __init__(self, data_path, batch_size=1, input_dims=(32, 128, 128, 3),
                 output_dims=(32, 128, 128, 1), shuffle=True, validation_split=0.2):
        """
        Args:
            data_path: Path to T2-weighted DICOM dataset directory
            batch_size: Batch size for training
            input_dims: Input dimensions (depth, height, width, channels)
            output_dims: Output dimensions (depth, height, width, channels)
            shuffle: Whether to shuffle data
            validation_split: Fraction of data to use for validation
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.shuffle = shuffle

        # Check for T2 and FLAIR folders with flexible naming
        possible_t2_names = ['T2', 't2', 'T2_images', 't2_images', 'T2W', 't2w']
        possible_flair_names = ['FLAIR', 'flair', 'FLAIR_images', 'flair_images', 'T2_FLAIR', 't2_flair']

        t2_folder_name = None
        flair_folder_name = None

        print(f"Checking for folders in: {data_path}")

        # List all items in the directory
        try:
            all_items = os.listdir(data_path)
            print(f"All items in directory: {all_items}")
            folders = [f for f in all_items if os.path.isdir(os.path.join(data_path, f))]
            print(f"Folders found: {folders}")

            # Find T2 folder
            for folder in folders:
                if any(t2_name in folder.upper() for t2_name in ['T2', 'T2W']):
                    if not any(flair_name in folder.upper() for flair_name in ['FLAIR']):
                        t2_folder_name = folder
                        print(f"Found T2 folder: {folder}")
                        break

            # Find FLAIR folder
            for folder in folders:
                if any(flair_name in folder.upper() for flair_name in ['FLAIR']):
                    flair_folder_name = folder
                    print(f"Found FLAIR folder: {folder}")
                    break

        except Exception as e:
            print(f"Error listing directory: {e}")

        if t2_folder_name and flair_folder_name:
            # Root-level organization: single patient with T2/FLAIR folders
            self.patient_folders = ['.']  # Single "patient" representing the root
            self.has_root_folders = True
            self.t2_folder_name = t2_folder_name
            self.flair_folder_name = flair_folder_name
            print(f"Using root-level folder organization: T2='{t2_folder_name}', FLAIR='{flair_folder_name}'")
        else:
            # Standard patient folder organization
            self.patient_folders = [f for f in os.listdir(data_path)
                                   if os.path.isdir(os.path.join(data_path, f))]
            self.patient_folders.sort()
            self.has_root_folders = False
            print(f"Using standard patient folder organization: {len(self.patient_folders)} folders")

        # Split into train/validation
        if len(self.patient_folders) == 1:
            # Special case: single patient, put in training
            self.train_patients = self.patient_folders
            self.val_patients = []
            print("Single patient dataset - using for training only")
        else:
            split_idx = int(len(self.patient_folders) * (1 - validation_split))
            # Ensure at least 1 training sample
            split_idx = max(1, min(split_idx, len(self.patient_folders) - 1))
            self.train_patients = self.patient_folders[:split_idx]
            self.val_patients = self.patient_folders[split_idx:]

        print(f"Found {len(self.patient_folders)} patient folders")
        print(f"Root-level T2/FLAIR folders: {self.has_root_folders}")
        print(f"Training patients: {len(self.train_patients)}")
        print(f"Validation patients: {len(self.val_patients)}")

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.train_patients) / self.batch_size))

    def __getitem__(self, index):
        # Get batch of patient IDs
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        patient_batch = [self.train_patients[k] for k in indexes]

        X, y = self.__data_generation(patient_batch)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.train_patients))
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
            ), order=1)
        else:  # 3D
            new_volume = zoom(volume, (
                target_shape[0] / volume.shape[0],
                target_shape[1] / volume.shape[1],
                target_shape[2] / volume.shape[2]
            ), order=1)
        return new_volume

    def resample_volume(self, volume, target_shape):
        """Resample volume to match target shape using interpolation"""
        # Use zoom for resampling
        zoom_factors = (
            target_shape[0] / volume.shape[0],
            target_shape[1] / volume.shape[1],
            target_shape[2] / volume.shape[2]
        )
        resampled = zoom(volume, zoom_factors, order=1)  # Linear interpolation
        return resampled

    def _adjust_volume_shape(self, volume, target_shape):
        """Adjust volume shape to exactly match target by cropping/padding"""
        adjusted = np.zeros(target_shape, dtype=volume.dtype)

        # Copy the overlapping region
        crop_depth = min(volume.shape[0], target_shape[0])
        crop_height = min(volume.shape[1], target_shape[1])
        crop_width = min(volume.shape[2], target_shape[2])

        adjusted[:crop_depth, :crop_height, :crop_width] = volume[:crop_depth, :crop_height, :crop_width]

        return adjusted

    def _force_exact_shape(self, volume, target_shape):
        """Force volume to exact target shape using zoom + crop/pad"""
        # First try to zoom to approximate shape
        current_shape = volume.shape
        if current_shape != target_shape:
            # Calculate zoom factors
            zoom_factors = np.array(target_shape) / np.array(current_shape)
            # Use order=1 for linear interpolation
            volume = zoom(volume, zoom_factors, order=1)

        # Now ensure exact shape with crop/pad
        if volume.shape != target_shape:
            volume = self._adjust_volume_shape(volume, target_shape)

        return volume

    def identify_t2_sequence_type(self, dicom_file):
        """Identify if DICOM file is T2 (non-weighted) or FLAIR (T2-weighted)"""
        try:
            # First check filename for sequence information (more reliable)
            filename = os.path.basename(dicom_file).upper()

            if 'FLAIR' in filename:
                return 'FLAIR'
            elif 'T2' in filename and 'FLAIR' not in filename:
                # Make sure it's not T2-weighted by checking for weighting indicators
                if any(weight_indicator in filename for weight_indicator in ['WEIGHTED', 'T2W', 'W']):
                    return 'UNKNOWN'  # This might be T2-weighted, skip for now
                else:
                    return 'T2'

            # Fallback to DICOM metadata
            import pydicom
            ds = pydicom.dcmread(dicom_file, stop_before_pixels=True)

            # Check various DICOM tags for sequence identification
            series_desc = getattr(ds, 'SeriesDescription', '').upper()
            protocol_name = getattr(ds, 'ProtocolName', '').upper()
            sequence_name = getattr(ds, 'SequenceName', '').upper()

            # Look for FLAIR (T2-weighted) indicators first
            if ('FLAIR' in series_desc or 'FLAIR' in protocol_name or
                'FLAIR' in sequence_name or 'T2_FLAIR' in series_desc or
                'T2 FLAIR' in series_desc):
                return 'FLAIR'

            # Look for T2 (non-weighted) indicators
            elif ('T2' in series_desc and 'FLAIR' not in series_desc and
                  'WEIGHTED' not in series_desc and 'T2W' not in series_desc or
                  'T2' in protocol_name and 'FLAIR' not in protocol_name and
                  'WEIGHTED' not in protocol_name and 'T2W' not in protocol_name or
                  'T2' in sequence_name and 'FLAIR' not in sequence_name and
                  'WEIGHTED' not in sequence_name and 'T2W' not in sequence_name):
                return 'T2'

            # Default to unknown
            return 'UNKNOWN'

        except Exception as e:
            print(f"Error reading DICOM metadata for {dicom_file}: {e}")
            return 'UNKNOWN'

    def group_t2_dicoms_by_sequence(self, patient_folder):
        """Group DICOM files by sequence type for T2-weighted dataset"""
        patient_path = os.path.join(self.data_path, patient_folder)

        # Check if there are T2 and FLAIR subfolders using detected names
        if hasattr(self, 't2_folder_name') and hasattr(self, 'flair_folder_name'):
            t2_folder = os.path.join(patient_path, self.t2_folder_name)
            flair_folder = os.path.join(patient_path, self.flair_folder_name)

            if os.path.exists(t2_folder) and os.path.exists(flair_folder):
                # Use folder-based organization
                return self._pair_files_by_folder(t2_folder, flair_folder)

        # Fallback to filename-based identification
        return self._pair_files_by_filename_identification(patient_path)

    def _pair_files_by_folder(self, t2_folder, flair_folder):
        """Pair T2 and FLAIR files from separate folders using last 9 characters"""
        import glob

        # Get all DICOM files from T2 folder
        t2_files = []
        for file in os.listdir(t2_folder):
            if file.lower().endswith('.dcm') or file.lower().endswith('.dicom'):
                t2_files.append(os.path.join(t2_folder, file))

        # Get all DICOM files from FLAIR folder
        flair_files = []
        for file in os.listdir(flair_folder):
            if file.lower().endswith('.dcm') or file.lower().endswith('.dicom'):
                flair_files.append(os.path.join(flair_folder, file))

        print(f"Found {len(t2_files)} T2 files in '{os.path.basename(t2_folder)}' and {len(flair_files)} FLAIR files in '{os.path.basename(flair_folder)}'")

        # Create mapping based on last 9 characters of filename
        t2_dict = {}
        for t2_file in t2_files:
            filename = os.path.basename(t2_file)
            key = filename[-13:-4] if len(filename) > 13 else filename  # Last 9 chars before extension
            t2_dict[key] = t2_file

        flair_dict = {}
        for flair_file in flair_files:
            filename = os.path.basename(flair_file)
            key = filename[-13:-4] if len(filename) > 13 else filename  # Last 9 chars before extension
            flair_dict[key] = flair_file

        # Pair files with matching keys
        paired_t2 = []
        paired_flair = []

        for key in t2_dict:
            if key in flair_dict:
                paired_t2.append(t2_dict[key])
                paired_flair.append(flair_dict[key])
                print(f"Paired: T2 {os.path.basename(t2_dict[key])} <-> FLAIR {os.path.basename(flair_dict[key])}")

        # Sort by the key for consistent ordering
        paired_data = sorted(zip(paired_t2, paired_flair), key=lambda x: self._get_sort_key(x[0]))
        paired_t2, paired_flair = zip(*paired_data) if paired_data else ([], [])

        print(f"Successfully paired {len(paired_t2)} T2-FLAIR pairs")
        return list(paired_t2), list(paired_flair)

    def _pair_files_by_filename_identification(self, patient_path):
        """Fallback method using filename identification"""
        import pydicom

        # Find all DICOM files
        dicom_files = []
        for root, dirs, files in os.walk(patient_path):
            for file in files:
                if file.lower().endswith('.dcm') or file.lower().endswith('.dicom'):
                    dicom_files.append(os.path.join(root, file))

        # Group by sequence
        t2_files = []
        flair_files = []

        for dicom_file in dicom_files:
            seq_type = self.identify_t2_sequence_type(dicom_file)
            print(f"DICOM {os.path.basename(dicom_file)}: {seq_type}")
            if seq_type == 'T2':
                t2_files.append(dicom_file)
            elif seq_type == 'FLAIR':
                flair_files.append(dicom_file)

        # Sort by instance number if available
        def sort_by_instance(files):
            file_data = []
            for f in files:
                try:
                    ds = pydicom.dcmread(f, stop_before_pixels=True)
                    instance_num = getattr(ds, 'InstanceNumber', 0)
                    file_data.append((instance_num, f))
                except:
                    file_data.append((0, f))
            file_data.sort(key=lambda x: x[0])
            return [f for _, f in file_data]

        t2_files = sort_by_instance(t2_files)
        flair_files = sort_by_instance(flair_files)

        return t2_files, flair_files

    def _get_sort_key(self, file_path):
        """Get sort key for consistent ordering"""
        filename = os.path.basename(file_path)
        # Use the last 9 characters before extension as sort key
        return filename[-13:-4] if len(filename) > 13 else filename

    def load_dicom_volume(self, dicom_files):
        """Load a 3D volume from DICOM files"""
        import pydicom
        slices = []
        for dicom_file in dicom_files:
            try:
                ds = pydicom.dcmread(dicom_file)
                slice_data = ds.pixel_array.astype(np.float32)
                slices.append(slice_data)
            except Exception as e:
                print(f"Error loading DICOM {dicom_file}: {e}")
                continue

        if not slices:
            raise ValueError("No valid DICOM slices found")

        volume = np.stack(slices, axis=0)
        return volume

    def __data_generation(self, patient_batch):
        """Generate batch of 3D data from T2-weighted DICOM files"""
        X = np.empty((self.batch_size, *self.input_dims))
        y = np.empty((self.batch_size, *self.output_dims))

        for n_, patient in enumerate(patient_batch):
            try:
                # Handle root-level organization
                if self.has_root_folders and patient == '.':
                    # Use the data_path directly as the patient folder
                    patient_folder = ''
                else:
                    patient_folder = patient

                # Get T2 and FLAIR files for this patient
                t2_files, flair_files = self.group_t2_dicoms_by_sequence(patient_folder)

                print(f"Patient {patient}: Found {len(t2_files)} T2 files, {len(flair_files)} FLAIR (T2-weighted) files")

                if not t2_files or not flair_files:
                    print(f"Skipping patient {patient}: Missing T2 or FLAIR sequences")
                    X[n_] = np.zeros(self.input_dims)
                    y[n_] = np.zeros(self.output_dims)
                    continue

                # Load volumes
                t2_volume = self.load_dicom_volume(t2_files)
                flair_volume = self.load_dicom_volume(flair_files)

                # Handle different shapes - resample to match if needed
                if t2_volume.shape != flair_volume.shape:
                    print(f"Shape mismatch - T2: {t2_volume.shape}, FLAIR: {flair_volume.shape}")
                    # Resample the smaller volume to match the larger one
                    if np.prod(t2_volume.shape) > np.prod(flair_volume.shape):
                        # T2 is larger, resample FLAIR to match T2
                        flair_volume = self.resample_volume(flair_volume, t2_volume.shape)
                    else:
                        # FLAIR is larger, resample T2 to match FLAIR
                        t2_volume = self.resample_volume(t2_volume, flair_volume.shape)
                    print(f"After resampling - T2: {t2_volume.shape}, FLAIR: {flair_volume.shape}")

                # Ensure minimum depth and consistent shapes
                min_depth = min(t2_volume.shape[0], flair_volume.shape[0])
                t2_volume = t2_volume[:min_depth]
                flair_volume = flair_volume[:min_depth]

                # Rescale to target dimensions
                target_t2_shape = self.input_dims[:3]
                target_flair_shape = self.output_dims[:3]

                # Ensure exact shape matching by padding/cropping
                t2_volume = self._force_exact_shape(t2_volume, target_t2_shape)
                flair_volume = self._force_exact_shape(flair_volume, target_flair_shape)

                # Normalize
                t2_volume = self.normalisation(t2_volume)
                flair_volume = self.normalisation(flair_volume)

                # Expand dimensions for channels
                if len(t2_volume.shape) == 3:
                    t2_volume = np.expand_dims(t2_volume, axis=-1)
                if len(flair_volume.shape) == 3:
                    flair_volume = np.expand_dims(flair_volume, axis=-1)

                # Replicate channels for RGB input if needed
                if self.input_dims[-1] > t2_volume.shape[-1]:
                    t2_volume = np.repeat(t2_volume, self.input_dims[-1] // t2_volume.shape[-1], axis=-1)

                # Final shape check
                expected_x_shape = self.input_dims
                expected_y_shape = self.output_dims

                if t2_volume.shape != expected_x_shape:
                    print(f"Final shape mismatch for X: got {t2_volume.shape}, expected {expected_x_shape}")
                    t2_volume = np.zeros(expected_x_shape)
                if flair_volume.shape != expected_y_shape:
                    print(f"Final shape mismatch for y: got {flair_volume.shape}, expected {expected_y_shape}")
                    flair_volume = np.zeros(expected_y_shape)

                X[n_] = t2_volume
                y[n_] = flair_volume

                print(f"Loaded patient {patient}: T2 shape {t2_volume.shape}, FLAIR (T2-weighted) shape {flair_volume.shape}")

            except Exception as e:
                print(f"Error processing patient {patient}: {e}")
                # Fill with zeros on error
                X[n_] = np.zeros(self.input_dims)
                y[n_] = np.zeros(self.output_dims)

        return X, y


class RemindDICOMDataGenerator(keras.utils.Sequence):
    """
    Data Generator for REMIND DICOM volumes - identifies T2 and T2-FLAIR sequences
    """

    def __init__(self, data_path, batch_size=1, input_dims=(32, 128, 128, 3),
                 output_dims=(32, 128, 128, 1), shuffle=True, validation_split=0.2):
        """
        Args:
            data_path: Path to REMIND dataset directory containing patient subfolders
            batch_size: Batch size for training
            input_dims: Input dimensions (depth, height, width, channels)
            output_dims: Output dimensions (depth, height, width, channels)
            shuffle: Whether to shuffle data
            validation_split: Fraction of data to use for validation
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.shuffle = shuffle

        # Find all patient folders
        self.patient_folders = [f for f in os.listdir(data_path)
                               if os.path.isdir(os.path.join(data_path, f))]
        self.patient_folders.sort()

        # Split into train/validation
        split_idx = int(len(self.patient_folders) * (1 - validation_split))
        self.train_patients = self.patient_folders[:split_idx]
        self.val_patients = self.patient_folders[split_idx:]

        print(f"Found {len(self.patient_folders)} patient folders")
        print(f"Training patients: {len(self.train_patients)}")
        print(f"Validation patients: {len(self.val_patients)}")

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.train_patients) / self.batch_size))

    def __getitem__(self, index):
        # Get batch of patient IDs
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        patient_batch = [self.train_patients[k] for k in indexes]

        X, y = self.__data_generation(patient_batch)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.train_patients))
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
            ), order=1)
        else:  # 3D
            new_volume = zoom(volume, (
                target_shape[0] / volume.shape[0],
                target_shape[1] / volume.shape[1],
                target_shape[2] / volume.shape[2]
            ), order=1)
        return new_volume

    def resample_volume(self, volume, target_shape):
        """Resample volume to match target shape using interpolation"""
        # Use zoom for resampling
        zoom_factors = (
            target_shape[0] / volume.shape[0],
            target_shape[1] / volume.shape[1],
            target_shape[2] / volume.shape[2]
        )
        resampled = zoom(volume, zoom_factors, order=1)  # Linear interpolation
        return resampled

    def _adjust_volume_shape(self, volume, target_shape):
        """Adjust volume shape to exactly match target by cropping/padding"""
        adjusted = np.zeros(target_shape, dtype=volume.dtype)

        # Copy the overlapping region
        crop_depth = min(volume.shape[0], target_shape[0])
        crop_height = min(volume.shape[1], target_shape[1])
        crop_width = min(volume.shape[2], target_shape[2])

        adjusted[:crop_depth, :crop_height, :crop_width] = volume[:crop_depth, :crop_height, :crop_width]

        return adjusted

    def _force_exact_shape(self, volume, target_shape):
        """Force volume to exact target shape using zoom + crop/pad"""
        # First try to zoom to approximate shape
        current_shape = volume.shape
        if current_shape != target_shape:
            # Calculate zoom factors
            zoom_factors = np.array(target_shape) / np.array(current_shape)
            # Use order=1 for linear interpolation
            volume = zoom(volume, zoom_factors, order=1)

        # Now ensure exact shape with crop/pad
        if volume.shape != target_shape:
            volume = self._adjust_volume_shape(volume, target_shape)

        return volume

    def identify_sequence_type(self, dicom_file):
        """Identify if DICOM file is T2 or T2-FLAIR based on metadata"""
        try:
            import pydicom
            ds = pydicom.dcmread(dicom_file, stop_before_pixels=True)

            # Check various DICOM tags for sequence identification
            series_desc = getattr(ds, 'SeriesDescription', '').upper()
            protocol_name = getattr(ds, 'ProtocolName', '').upper()
            sequence_name = getattr(ds, 'SequenceName', '').upper()

            # Look for FLAIR indicators first (more specific)
            if ('FLAIR' in series_desc or 'FLAIR' in protocol_name or
                'FLAIR' in sequence_name or 'T2_FLAIR' in series_desc or
                'T2 FLAIR' in series_desc):
                return 'FLAIR'

            # Look for T2 indicators
            elif ('T2' in series_desc and 'FLAIR' not in series_desc or
                  'T2' in protocol_name and 'FLAIR' not in protocol_name or
                  'T2' in sequence_name and 'FLAIR' not in sequence_name):
                return 'T2'

            # Default to unknown
            return 'UNKNOWN'

        except Exception as e:
            print(f"Error reading DICOM metadata for {dicom_file}: {e}")
            return 'UNKNOWN'

    def group_dicoms_by_sequence(self, patient_folder):
        """Group DICOM files by sequence type for a patient"""
        import pydicom
        patient_path = os.path.join(self.data_path, patient_folder)

        # Find all DICOM files
        dicom_files = []
        for root, dirs, files in os.walk(patient_path):
            for file in files:
                if file.lower().endswith('.dcm') or file.lower().endswith('.dicom'):
                    dicom_files.append(os.path.join(root, file))

        # Group by sequence
        t2_files = []
        flair_files = []

        for dicom_file in dicom_files:
            seq_type = self.identify_sequence_type(dicom_file)
            if seq_type == 'T2':
                t2_files.append(dicom_file)
            elif seq_type == 'FLAIR':
                flair_files.append(dicom_file)

        # Sort by instance number if available
        def sort_by_instance(files):
            file_data = []
            for f in files:
                try:
                    ds = pydicom.dcmread(f, stop_before_pixels=True)
                    instance_num = getattr(ds, 'InstanceNumber', 0)
                    file_data.append((instance_num, f))
                except:
                    file_data.append((0, f))
            file_data.sort(key=lambda x: x[0])
            return [f for _, f in file_data]

        t2_files = sort_by_instance(t2_files)
        flair_files = sort_by_instance(flair_files)

        return t2_files, flair_files

    def load_dicom_volume(self, dicom_files):
        """Load a 3D volume from DICOM files"""
        import pydicom
        slices = []
        for dicom_file in dicom_files:
            try:
                ds = pydicom.dcmread(dicom_file)
                slice_data = ds.pixel_array.astype(np.float32)
                slices.append(slice_data)
            except Exception as e:
                print(f"Error loading DICOM {dicom_file}: {e}")
                continue

        if not slices:
            raise ValueError("No valid DICOM slices found")

        volume = np.stack(slices, axis=0)
        return volume

    def __data_generation(self, patient_batch):
        """Generate batch of 3D data from DICOM files"""
        X = np.empty((self.batch_size, *self.input_dims))
        y = np.empty((self.batch_size, *self.output_dims))

        for n_, patient in enumerate(patient_batch):
            try:
                # Get T2 and FLAIR files for this patient
                t2_files, flair_files = self.group_dicoms_by_sequence(patient)

                print(f"Patient {patient}: Found {len(t2_files)} T2 files, {len(flair_files)} FLAIR files")

                if not t2_files or not flair_files:
                    print(f"Skipping patient {patient}: Missing T2 or FLAIR sequences")
                    X[n_] = np.zeros(self.input_dims)
                    y[n_] = np.zeros(self.output_dims)
                    continue

                # Load volumes
                t2_volume = self.load_dicom_volume(t2_files)
                flair_volume = self.load_dicom_volume(flair_files)

                # Handle different shapes - resample to match if needed
                if t2_volume.shape != flair_volume.shape:
                    print(f"Shape mismatch - T2: {t2_volume.shape}, FLAIR: {flair_volume.shape}")
                    # Resample the smaller volume to match the larger one
                    if np.prod(t2_volume.shape) > np.prod(flair_volume.shape):
                        # T2 is larger, resample FLAIR to match T2
                        flair_volume = self.resample_volume(flair_volume, t2_volume.shape)
                    else:
                        # FLAIR is larger, resample T2 to match FLAIR
                        t2_volume = self.resample_volume(t2_volume, flair_volume.shape)
                    print(f"After resampling - T2: {t2_volume.shape}, FLAIR: {flair_volume.shape}")

                # Ensure minimum depth and consistent shapes
                min_depth = min(t2_volume.shape[0], flair_volume.shape[0])
                t2_volume = t2_volume[:min_depth]
                flair_volume = flair_volume[:min_depth]

                # Rescale to target dimensions
                target_t2_shape = self.input_dims[:3]
                target_flair_shape = self.output_dims[:3]

                # Ensure exact shape matching by padding/cropping
                t2_volume = self._force_exact_shape(t2_volume, target_t2_shape)
                flair_volume = self._force_exact_shape(flair_volume, target_flair_shape)

                # Normalize
                t2_volume = self.normalisation(t2_volume)
                flair_volume = self.normalisation(flair_volume)

                # Expand dimensions for channels
                if len(t2_volume.shape) == 3:
                    t2_volume = np.expand_dims(t2_volume, axis=-1)
                if len(flair_volume.shape) == 3:
                    flair_volume = np.expand_dims(flair_volume, axis=-1)

                # Replicate channels for RGB input if needed
                if self.input_dims[-1] > t2_volume.shape[-1]:
                    t2_volume = np.repeat(t2_volume, self.input_dims[-1] // t2_volume.shape[-1], axis=-1)

                # Final shape check
                expected_x_shape = self.input_dims
                expected_y_shape = self.output_dims

                if t2_volume.shape != expected_x_shape:
                    print(f"Final shape mismatch for X: got {t2_volume.shape}, expected {expected_x_shape}")
                    t2_volume = np.zeros(expected_x_shape)
                if flair_volume.shape != expected_y_shape:
                    print(f"Final shape mismatch for y: got {flair_volume.shape}, expected {expected_y_shape}")
                    flair_volume = np.zeros(expected_y_shape)

                X[n_] = t2_volume
                y[n_] = flair_volume

                print(f"Loaded patient {patient}: T2 shape {t2_volume.shape}, FLAIR shape {flair_volume.shape}")

            except Exception as e:
                print(f"Error processing patient {patient}: {e}")
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


class NiftiBraTSDataGenerator(keras.utils.Sequence):
    """
    Data Generator for BraTS NIfTI volumes - T2 as input, FLAIR as target for fat suppression
    """

    def __init__(self, data_path, batch_size=1, input_dims=(32, 128, 128, 3),
                 output_dims=(32, 128, 128, 1), shuffle=True, validation_split=0.2):
        """
        Args:
            data_path: Path to BraTS dataset directory containing patient subfolders
            batch_size: Batch size for training
            input_dims: Input dimensions (depth, height, width, channels)
            output_dims: Output dimensions (depth, height, width, channels)
            shuffle: Whether to shuffle data
            validation_split: Fraction of data to use for validation
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.shuffle = shuffle

        # Find all patient folders
        self.patient_folders = [f for f in os.listdir(data_path)
                               if os.path.isdir(os.path.join(data_path, f))]
        self.patient_folders.sort()

        # Split into train/validation
        split_idx = int(len(self.patient_folders) * (1 - validation_split))
        self.train_patients = self.patient_folders[:split_idx]
        self.val_patients = self.patient_folders[split_idx:]

        print(f"Found {len(self.patient_folders)} patient folders")
        print(f"Training patients: {len(self.train_patients)}")
        print(f"Validation patients: {len(self.val_patients)}")

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.train_patients) / self.batch_size))

    def __getitem__(self, index):
        # Get batch of patient IDs
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        patient_batch = [self.train_patients[k] for k in indexes]

        X, y = self.__data_generation(patient_batch)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.train_patients))
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
            ), order=1)
        else:  # 3D
            new_volume = zoom(volume, (
                target_shape[0] / volume.shape[0],
                target_shape[1] / volume.shape[1],
                target_shape[2] / volume.shape[2]
            ), order=1)
        return new_volume

    def load_nifti_volume(self, patient_folder, modality):
        """Load NIfTI volume for a patient and modality"""
        # Construct filename pattern (e.g., BraTS2021_00000_flair.nii.gz)
        patient_id = os.path.basename(patient_folder)

        # Try different extensions
        patterns = [
            os.path.join(self.data_path, patient_folder, f"{patient_id}_{modality}.nii.gz"),
            os.path.join(self.data_path, patient_folder, f"{patient_id}_{modality}.nii"),
            os.path.join(self.data_path, patient_folder, f"{patient_id}_{modality.upper()}.nii.gz"),
            os.path.join(self.data_path, patient_folder, f"{patient_id}_{modality.upper()}.nii"),
        ]

        pattern = None
        for p in patterns:
            if os.path.exists(p):
                pattern = p
                break

        if pattern is None:
            raise FileNotFoundError(f"NIfTI file not found for {patient_id}_{modality}, tried: {patterns}")

        # Load NIfTI
        nii_img = nib.load(pattern)
        volume = nii_img.get_fdata()

        # Ensure volume is float32
        volume = volume.astype(np.float32)

        return volume

    def __data_generation(self, patient_batch):
        """Generate batch of 3D data from NIfTI files"""
        X = np.empty((self.batch_size, *self.input_dims))
        y = np.empty((self.batch_size, *self.output_dims))

        for n_, patient in enumerate(patient_batch):
            try:
                # Load T2 as input (fat image)
                t2_volume = self.load_nifti_volume(patient, 't2')

                # Load FLAIR as target (suppressed image)
                flair_volume = self.load_nifti_volume(patient, 'flair')

                # Ensure same shape (BraTS volumes should be aligned)
                assert t2_volume.shape == flair_volume.shape, f"Shape mismatch: T2 {t2_volume.shape} vs FLAIR {flair_volume.shape}"

                # Rescale to target dimensions
                t2_volume = self.rescale_3d(t2_volume, self.input_dims[:3])
                flair_volume = self.rescale_3d(flair_volume, self.output_dims[:3])

                # Normalize
                t2_volume = self.normalisation(t2_volume)
                flair_volume = self.normalisation(flair_volume)

                # Expand dimensions for channels
                if len(t2_volume.shape) == 3:
                    t2_volume = np.expand_dims(t2_volume, axis=-1)
                if len(flair_volume.shape) == 3:
                    flair_volume = np.expand_dims(flair_volume, axis=-1)

                # Replicate channels for RGB input if needed
                if self.input_dims[-1] > t2_volume.shape[-1]:
                    t2_volume = np.repeat(t2_volume, self.input_dims[-1] // t2_volume.shape[-1], axis=-1)

                X[n_] = t2_volume
                y[n_] = flair_volume

                print(f"Loaded patient {patient}: T2 shape {t2_volume.shape}, FLAIR shape {flair_volume.shape}")

            except Exception as e:
                print(f"Error processing patient {patient}: {e}")
                # Fill with zeros on error
                X[n_] = np.zeros(self.input_dims)
                y[n_] = np.zeros(self.output_dims)

        return X, y
