#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from errno import EEXIST
from os import makedirs, path
import os
import OpenEXR
import Imath
import numpy as np
from PIL import Image

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)

def read_exr_image(file_path):
    exr_file = OpenEXR.InputFile(file_path)
    dw = exr_file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    float_type = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = exr_file.header()['channels'].keys()
    
    # Print available channels for debugging

    # Try 'Z' channel first, fallback to 'R' if 'Z' is not available
    if 'Z' in channels:
        channel = exr_file.channel('Z', float_type)
    elif 'R' in channels:
        channel = exr_file.channel('R', float_type)
    else:
        raise ValueError(f"No suitable depth channel found in {file_path}. Available channels: {channels}")

    depth_map = np.frombuffer(channel, dtype=np.float32)
    depth_map = np.array(depth_map)  # Copy array to ensure it's resizable
    depth_map = np.reshape(depth_map, (size[1], size[0]))

    # Normalize depth values to 0-255 for visualization
    depth_min = np.min(depth_map)
    depth_max = np.max(depth_map)
    depth_map = 255 * (depth_map - depth_min) / (depth_max - depth_min)
    depth_map = depth_map.astype(np.uint8)
    depth_image = Image.fromarray(depth_map)

    return depth_image