# Image Analysis and Simulation Pipeline
This repository contains scripts and tools for a comprehensive pipeline to analyze fluorescent microscopy images, generate simulations, and perform sensitivity analysis. Below is an overview of the included files and a step-by-step guide to using the pipeline.

## Repository Contents
README.md: Documentation for understanding and using the pipeline.
## SSA_simulation.py: 
Script for running simulations using a Stochastic Simulation Algorithm (SSA).
## background_edit.py: 
Script to preprocess and edit the background of microscopy images.
## contour_mask.py: 
Script for segmenting images and generating contour masks.
## extract_pixels.py: 
Script for extracting pixel values from the processed images.
## mask_usage.py: 
Script to apply the generated masks on images.
## Extract_pixels: 
Directory or module for storing or handling pixel data extracted in the pipeline.
## Workflow Steps
Step 1: Background Editing (background_edit.py)
This step involves preprocessing the raw fluorescent microscopy images. The goal is to edit and standardize the background of the images to ensure consistency for subsequent analysis.
In our case, the input images include: Green channel(endosome),Red channel(siRNA).

Step 2: Contour Mask Creation (contour_mask.py)
Here, raw microscopy images containing multiple fluorescent channels (red, green, blue, and phase images) are processed for segmentation.
The blue channel (nucleus) is retained during this step to accurately define the boundaries of cellular clusters. Contour masks are created to segment and outline the areas of interest.

Step 3: Mask Application (mask_usage.py)
Using the masks generated in Step 2, we create separate masks for each image. Since cells are dynamic and move across images, a unique mask is generated for each frame. These masks are then overlaid on the preprocessed images from Step 1. The intersecting regions are saved for further analysis.

Step 4: Extracting Pixels (extract_pixels.py)
This step extracts pixel-level information from the intersected regions of the images saved in Step 3. The pixel data is essential for further computational modeling and analysis.

Step 5: SSA Simulation (SSA_simulation.py)
Simulations are performed using a Stochastic Simulation Algorithm (SSA) model. This step models the biological processes and interactions captured in the microscopy images. Sensitivity analysis using Sobol and FAST (Fourier Amplitude Sensitivity Test) methods. These techniques help identify key parameters and assess their impact on the simulation outputs.

# How to Use
1) Preprocess images using background_edit.py to edit the backgrounds.
2) Segment and create masks using contour_mask.py for precise cell boundary definitions.
3) Apply the masks to images with mask_usage.py and save the intersected regions.
4) Extract pixel-level data using extract_pixels.py for computational analysis.
5) Run simulations using SSA_simulation.py to model biological interactions.
6) Perform sensitivity analysis using the included methods to identify critical parameters.

# Requirements
To run this pipeline, you need the following:
1) Python 3.8.19
2) Required libraries: numpy, matplotlib, scipy, pandas, opencv-python and more information on version in available in package_version.py file.
3) Image processing tools compatible with microscopy images
