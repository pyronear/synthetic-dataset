# Copyright (C) 2019-2022, Pyronear.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from syntheticdataset.randomization.depth_estimation import detect_sky_from_depth
import random
import numpy as np


def get_random_start_point(src_image):
    """
    This methods generates a random dx index as a start point for the smoke
    and then leverages sky detection to better define which dy to choose

    Args :
        src_image (np.array) : source image

    Returns :
        tuple start point of the smoke
    """

    # Define the dx value to start the smoke at
    img_width = src_image.shape[1]
    dx = random.randint(0, img_width - 1)

    # Detect the sky in the source image
    sky_mask = detect_sky_from_depth(src_image)

    # Search for the max dy value not to be in the sky at the dx index
    max_dy = np.argmax(sky_mask[:, dx] == 0) - 1
    min_dy = np.argmin(sky_mask[:, dx] == 0) - 1
    dy = random.randint(min_dy, max_dy - 1)

    return dx, dy
