#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure

# Function to adjust brightness and contrast and return combined image
def adjust_and_combine(green, red, g_in_min, g_in_max, g_out_min, g_out_max, r_in_min, r_in_max, r_out_min, r_out_max):
    # Ensure green and red are single-channel (2D arrays)
    if green.ndim == 3:
        green = green[:, :, 1]  # Extract green channel if 3D
    if red.ndim == 3:
        red = red[:, :, 0]  # Extract red channel if 3D

    # Adjust brightness for green channel
    brightened_green = exposure.rescale_intensity(
        green, in_range=(g_in_min, g_in_max), out_range=(g_out_min, g_out_max)
    )
    # Adjust contrast for red channel
    contrast_adjusted_red = exposure.rescale_intensity(
        red, in_range=(r_in_min, r_in_max), out_range=(r_out_min, r_out_max)
    )

    # Combine adjusted channels (setting blue to zero)
    combined_image = np.zeros((green.shape[0], green.shape[1], 3))  # Ensure correct shape
    combined_image[:, :, 1] = brightened_green  # Assign adjusted green channel
    combined_image[:, :, 0] = contrast_adjusted_red  # Assign adjusted red channel
    return combined_image

# Directories
input_dir = "C:\\Users\\fnisha\\Box\\AB-Cook-Nisha Research\\unedited_ovary\\80"  # Replace with your input directory
output_dir = "80_background_edit"  # Replace with your output directory

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Parameters for brightness and contrast adjustment
g_in_min, g_in_max, g_out_min, g_out_max = 0, 0.24, 0, 0.84  # these values are chosen with lab researchers based on their requirement
r_in_min, r_in_max, r_out_min, r_out_max = 0.1, 0.6, 0.0, 1 # these values are chosen with lab researchers based on their requirement

# Iterate through subfolders
for subfolder in os.listdir(input_dir):
    subfolder_path = os.path.join(input_dir, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    # Create corresponding output subfolder
    output_subfolder_path = os.path.join(output_dir, subfolder)
    os.makedirs(output_subfolder_path, exist_ok=True)

    # Initialize placeholders for channel images
    green_image = None
    red_image = None

    # Select channel 2 and channel 3 images
    for file_name in os.listdir(subfolder_path):
        if "_XY01_Z002_CH3" in file_name:  # Green channel (CH3)
            green_image = io.imread(os.path.join(subfolder_path, file_name))
        elif "_XY01_Z002_CH2" in file_name:  # Red channel (CH2)
            red_image = io.imread(os.path.join(subfolder_path, file_name))

        # If both channels are found, process them
        if green_image is not None and red_image is not None:
            # Normalize channels
            if green_image.max() > 1:
                green_image = green_image / 255.0
            if red_image.max() > 1:
                red_image = red_image / 255.0

            # Apply adjustments
            combined_image = adjust_and_combine(
                green_image,
                red_image,
                g_in_min, g_in_max, g_out_min, g_out_max,
                r_in_min, r_in_max, r_out_min, r_out_max
            )

            # Save combined image to the output directory
            output_file_name = f"{subfolder}_combined.tif"
            output_file_path = os.path.join(output_subfolder_path, output_file_name)
            io.imsave(output_file_path, (combined_image * 255).astype(np.uint8))
            print(f"Processed and saved: {output_file_path}")

            # Reset images for the next pair
            green_image = None
            red_image = None




