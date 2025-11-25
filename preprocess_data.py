# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 21:46:15 2024

@author: amraa

Fat Suppression Data Preprocessing Script

This script preprocesses raw fat suppression MRI images by:
1. Scanning subfolders for paired 'fat' and 'suppression' images
2. Resizing images to 512x512 resolution
3. Balancing pairs and saving processed images for training

Developer Notes:
- Expects 'raw images/*' directory with subfolders containing paired files
- Files must contain 'fat' and 'suppression' keywords in filenames
- Outputs saved to 'processed/fat/' and 'processed/suppressed/' folders
- Randomly displays 10% of processed pairs as visual verification
"""

import numpy as np
import matplotlib.pyplot as plt  # type: ignore
from scipy.ndimage import zoom
import glob
import skimage
from tqdm import tqdm
import re
import os

# Directory paths - raw images should be in subfolders under 'raw images'
# Filenames should contain 'fat' or 'suppression' keywords to be detected
img_folder = 'raw images/*'
save_fold = 'processed/'

# Create output directories if they don't exist
os.makedirs(f'{save_fold}/fat', exist_ok=True)
os.makedirs(f'{save_fold}/suppressed', exist_ok=True)

# Initialize empty lists for potential global use (though not used here)
fat = []
suppressed = []

# Global counter for unique file numbering across subfolders
i = 0

# Debug: Check if input directories exist and contain data
print(f"Scanning for subfolders in: {img_folder}")
subfolders = glob.glob(img_folder)
print(f"Found {len(subfolders)} subfolders: {subfolders}")

if len(subfolders) == 0:
    print("ERROR: No 'raw images' folder or subfolders found!")
    print("Please ensure your data is in a 'raw images' folder with subfolders containing files.")
    exit(1)

# Main processing loop for each subfolder
for subfold in tqdm(subfolders, total=len(subfolders)):
    # Temporary lists for current subfolder
    fat_ = []
    suppressed_ = []

    # Scan all files in current subfolder and classify by naming
    for file in glob.glob(subfold + '/*'):
        normalized_name = file.lower()
        if 'fat' in normalized_name:
            fat_.append(skimage.io.imread(file))
        if 'suppression' in normalized_name:
            suppressed_.append(skimage.io.imread(file))

    # Determine minimum length to ensure balanced pairs
    print('Hello  Processing Subfolder:', subfold)
    min_len = min(len(fat_), len(suppressed_))

    print('Hello  Min length:', len(fat_), len(suppressed_), '-> Using:', min_len)

    # Skip subfolders with no paired files to avoid division by zero errors
    if min_len == 0:
        continue

    
    # Convert lists to ndarray and resize to target dimensions
    fat_ar = np.array(fat_)
    fat_ar = zoom(fat_ar, (min_len / fat_ar.shape[0],  # Select first min_len slices
                          512 / fat_ar.shape[1],      # Resize width to 512
                          512 / fat_ar.shape[2],      # Resize height to 512
                          1))                          # Maintain single channel

    # Same process for suppressed images
    suppressed_ar = np.array(suppressed_)
    suppressed_ar = zoom(suppressed_ar, (min_len / suppressed_ar.shape[0],
                                         512 / suppressed_ar.shape[1],
                                         512 / suppressed_ar.shape[2],
                                         1))

    print(fat_ar.__len__(), suppressed_ar.__len__())
    # Save each paired slice with zero-padded sequential naming
    for im_fat, im_sup in zip(fat_ar, suppressed_ar):
        i += 1
        file_name = re.sub(r'^(\d+)$', lambda m: m.group(1).zfill(6), str(i))
        print(f'Processing and saving file: {file_name}.jpg')

        # Visual verification: randomly show 10% of processed pairs
        if np.random.random(1) < 0.1:
            plt.suptitle(f'Example {file_name}')
            plt.subplot(1, 2, 1)
            plt.imshow(im_fat, cmap='gray')
            plt.title('Fat')
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.title('Suppressed')
            plt.imshow(im_sup, cmap='gray')
            plt.axis('off')
            plt.tight_layout()
            plt.show()  # This will display during processing

        # Save processed images to respective folders
        skimage.io.imsave(f'{save_fold}/suppressed/{file_name}.jpg', im_sup)
        skimage.io.imsave(f'{save_fold}/fat/{file_name}.jpg', im_fat)
