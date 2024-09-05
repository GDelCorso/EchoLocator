#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the EcoLocator package, which provides a normalization strategy
of ecographic images.
"""

#%% Import the package:
import EcoLocator

#%% Define the local input path
input_path = 'InputEco'

#%% Initialize the class
my_EcoLoc = EcoLocator.EcoLoc(input_path)


#%% Apply the pre-process removal
im_temp_np = my_EcoLoc.pre_remove()


#%% Apply the method which removes the watermarks
my_EcoLoc.no_watermark()

#%% Cropping method to crop and center the image
my_EcoLoc.standard_crops(resolution_output = 360)

#%% Square crops using polar coordinates
my_EcoLoc.square_crops(resolution_output=360)