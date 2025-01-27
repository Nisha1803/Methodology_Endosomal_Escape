#!/usr/bin/env python
# coding: utf-8

from __future__ import division, print_function, absolute_import
from importnb import Notebook
with Notebook():
    import contour_mask

from contour_mask import interect, create_mask
import os
from pathlib import Path
from skimage.io import imread, imsave
import numpy as np
from contour_mask import interect, create_mask  # Import your processing functions


# # for edited images
# Define input and output directories for 80
input_dir1 = "C:\\Users\\fnisha\\Box\\AB-Cook-Nisha Research\\unedited_ovary\\80"  # for raw image
input_dir2 = "C:\\Users\\fnisha\\Box\\AB-Cook-Nisha Research\\data_ovary\\80.0" #for edited image
input_dir3 = "80_background_edit" #for edited image
output_dir1 = "80\\80_raw_image_output_folder"  # output for raw image
output_dir2 = "80\\80_edited_images"  # output for edited
output_dir3 = "80\\80_background_edit"  # output for edited
mask_output_dir = "80_mask"


# In[68]:


import matplotlib.pyplot as plt

print("Generating masks from:", input_dir1)
for folder_name in os.listdir(input_dir1):
    input_folder = os.path.join(input_dir1, folder_name)

    if not os.path.isdir(input_folder):
        continue

    for file_name in os.listdir(input_folder):
        # if file_name.endswith("_Z001_Overlay.tif"): #for 40
        if file_name.endswith("_Z002_Overlay.tif"): #for 80
            image_path = os.path.join(input_folder, file_name)

            try:
                # Load image and generate mask
                image = imread(image_path)
                print(f"Generating mask for: {image_path}")
                # if image.dtype != 'float32':
                #     image = image / 255.0  # Ensure image is in the [0, 1] range

                # # Step 1: Brightness adjustment
                # image = exposure.rescale_intensity(image, in_range=(0.4, 1), out_range=(0.1, 0.9))
                # # image=image*255
                mask = interect(image, 100)  # Generate mask interactively
            except Exception as e:
                print(f"Error generating mask for {image_path}: {e}")
                continue

            # Overlay the mask on the original image
            overlay_image = image.copy()
            if mask.ndim == 2:  # Ensure mask is 2D
                overlay_image[:, :, 2] = np.maximum(overlay_image[:, :, 2], mask)  # Highlight mask on red channel

            # Plot the original image, mask, and overlay
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(image)
            ax[0].set_title("Original Image")

            ax[1].imshow(mask, cmap="gray")
            ax[1].set_title("Generated Mask")

            ax[2].imshow(overlay_image)
            ax[2].set_title("Overlay: Mask on Original")

            for a in ax:
                a.axis("off")
            plt.show()

            # Save the mask in the same folder name as input_dir1
            mask_folder = os.path.join(mask_output_dir, folder_name)
            os.makedirs(mask_folder, exist_ok=True)
            mask_output_path = os.path.join(mask_folder, file_name)

            try:
                imsave(mask_output_path, mask.astype(np.uint8))
                print(f"Mask saved to: {mask_output_path}")
            except Exception as e:
                print(f"Error saving mask to {mask_output_path}: {e}")


# In[76]:


# Step 2: Apply Masks to Input Directory 2
print("Applying masks to:", input_dir2)
mask_output_dir ='80_mask'
for folder_name in os.listdir(input_dir2):  # Iterate through folders in input_dir2
    input_folder = os.path.join(input_dir2, folder_name)

    if not os.path.isdir(input_folder):  # Skip non-directory entries
        continue

    # Locate the corresponding mask folder based on folder name
    mask_folder = os.path.join(mask_output_dir, folder_name)
    if not os.path.isdir(mask_folder):
        print(f"Mask folder not found for {folder_name}, skipping.")
        continue

    # Get the mask file from the mask folder
    mask_files = os.listdir(mask_folder)
    if len(mask_files) != 1:  # Ensure exactly one mask file exists
        print(f"Expected one mask in {mask_folder}, found {len(mask_files)}. Skipping.")
        continue

    mask_path = os.path.join(mask_folder, mask_files[0])  # Get the single mask file

    for file_name in os.listdir(input_folder):  # Iterate through files in the input folder
        # if file_name.endswith("Red and Green Overlay.tif"):  # Match image files for 40
        if file_name.endswith("Red and Green Overlay.tif"): # Match image files for 80
            
            image_path = os.path.join(input_folder, file_name)

            try:
                # Load the image and the mask
                image = imread(image_path)
                mask = imread(mask_path)
                print(f"Applying mask from {mask_path} to {image_path}")

                # Apply the mask
                processed_image = create_mask(image, mask)
            except Exception as e:
                print(f"Error applying mask to {image_path}: {e}")
                continue

            # Save the processed image in the output directory
            output_folder = os.path.join(output_dir2, folder_name)
            os.makedirs(output_folder, exist_ok=True)
            output_image_path = os.path.join(output_folder, file_name)

            try:
                imsave(output_image_path, processed_image.astype(np.uint8))
                print(f"Processed image saved to: {output_image_path}")
            except Exception as e:
                print(f"Error saving processed image to {output_image_path}: {e}")


# # for raw images

# In[11]:


import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, restoration, exposure
from scipy.ndimage import gaussian_filter
# Step 2: Apply Masks to Input Directory 1
print("Applying masks to:", input_dir1)

for folder_name in os.listdir(input_dir1):  # Iterate through folders in input_dir2
    input_folder = os.path.join(input_dir1, folder_name)

    if not os.path.isdir(input_folder):  # Skip non-directory entries
        continue

    # Locate the corresponding mask folder based on folder name
    mask_folder = os.path.join(mask_output_dir, folder_name)
    if not os.path.isdir(mask_folder):
        print(f"Mask folder not found for {folder_name}, skipping.")
        continue

    # Get the mask file from the mask folder
    mask_files = os.listdir(mask_folder)
    if len(mask_files) != 1:  # Ensure exactly one mask file exists
        print(f"Expected one mask in {mask_folder}, found {len(mask_files)}. Skipping.")
        continue

    mask_path = os.path.join(mask_folder, mask_files[0])  # Get the single mask file

    for file_name in os.listdir(input_folder):  # Iterate through files in the input folder
        
        # if file_name.endswith("XY01_Z001_Overlay.tif"):  # Match image files for 40
        if file_name.endswith("XY01_Z002_Overlay.tif"):  # Match image files for 40
            image_path = os.path.join(input_folder, file_name)

            try:
                # Load the image and the mask
                image = imread(image_path)
                # if image.dtype != 'float32':
                #     image = image / 255.0  # Ensure image is in the [0, 1] range

                # # Step 1: Brightness adjustment
                # image = exposure.rescale_intensity(image, in_range=(0.4, 1), out_range=(0.1, 0.9))
                # image=image*255.0
                mask = imread(mask_path)
                print(f"Applying mask from {mask_path} to {image_path}")

                # Apply the mask
                processed_image = create_mask(image, mask)
            except Exception as e:
                print(f"Error applying mask to {image_path}: {e}")
                continue

            # Save the processed image in the output directory
            output_folder = os.path.join(output_dir1, folder_name)
            os.makedirs(output_folder, exist_ok=True)
            output_image_path = os.path.join(output_folder, file_name)

            try:
                imsave(output_image_path, processed_image.astype(np.uint8))
                print(f"Processed image saved to: {output_image_path}")
            except Exception as e:
                print(f"Error saving processed image to {output_image_path}: {e}")


# # for background edit image

# In[41]:


import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, restoration, exposure
from scipy.ndimage import gaussian_filter
# Step 2: Apply Masks to Input Directory 1
print("Applying masks to:", input_dir3)

for folder_name in os.listdir(input_dir3):  # Iterate through folders in input_dir2
    input_folder = os.path.join(input_dir3, folder_name)

    if not os.path.isdir(input_folder):  # Skip non-directory entries
        continue

    # Locate the corresponding mask folder based on folder name
    mask_folder = os.path.join(mask_output_dir, folder_name)
    if not os.path.isdir(mask_folder):
        print(f"Mask folder not found for {folder_name}, skipping.")
        continue

    # Get the mask file from the mask folder
    mask_files = os.listdir(mask_folder)
    if len(mask_files) != 1:  # Ensure exactly one mask file exists
        print(f"Expected one mask in {mask_folder}, found {len(mask_files)}. Skipping.")
        continue

    mask_path = os.path.join(mask_folder, mask_files[0])  # Get the single mask file

    for file_name in os.listdir(input_folder):  # Iterate through files in the input folder
        
        # if file_name.endswith("XY01_Z001_Overlay.tif"):  # Match image files for 40
        if file_name.endswith("combined.tif"):  # Match image files for 80
            image_path = os.path.join(input_folder, file_name)

            try:
                # Load the image and the mask
                image = imread(image_path)
                # if image.dtype != 'float32':
                #     image = image / 255.0  # Ensure image is in the [0, 1] range

                # # Step 1: Brightness adjustment
                # image = exposure.rescale_intensity(image, in_range=(0.4, 1), out_range=(0.1, 0.9))
                # image=image*255.0
                mask = imread(mask_path)
                print(f"Applying mask from {mask_path} to {image_path}")

                # Apply the mask
                processed_image = create_mask(image, mask)
            except Exception as e:
                print(f"Error applying mask to {image_path}: {e}")
                continue

            # Save the processed image in the output directory
            output_folder = os.path.join(output_dir3, folder_name)
            os.makedirs(output_folder, exist_ok=True)
            output_image_path = os.path.join(output_folder, file_name)

            try:
                imsave(output_image_path, processed_image.astype(np.uint8))
                print(f"Processed image saved to: {output_image_path}")
            except Exception as e:
                print(f"Error saving processed image to {output_image_path}: {e}")


