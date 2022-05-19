# Copyright (C) 2019-2022, Pyronear.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import cv2
import numpy as np
from syntheticdataset.poisson_blending_utils import create_mask, poisson_blend


def basic_blending(img, smoke, offset=(0, 0), opacity=0.8):
    """Add smoke on image using basic image blending

    Args:
        img (np.array): background image
        smoke (np.array): smoke image
        offset (tuple, optional): smoke location offset (dy, dx). Defaults to (0, 0).
        opacity (float, optional): smoke image opacity in [0, 1]. Defaults to (0, 0).

    Returns:
        np.array: result image
        np.array: result mask
    """

    ks = 7
    kernel = np.ones((ks, ks), np.float32) / (ks**2)

    dy, dx = offset
    temp = img[dy : dy + smoke.shape[0], dx : dx + smoke.shape[1], :]
    dst = cv2.filter2D(smoke, -1, kernel)
    mask_dst = dst[:, :, 0] > 50
    alpha = 1 - opacity * dst / np.max(dst)
    res = temp * alpha + smoke[:, :, ::-1] * (1 - alpha)
    img[dy : dy + smoke.shape[0], dx : dx + smoke.shape[1], :] = res
    mask = img[:, :, 0] * 0
    mask[dy : dy + smoke.shape[0], dx : dx+ smoke.shape[1]] = mask_dst

    return img, mask


def poisson_blending(img, smoke, offset=(0, 0)):
    """Add smoke on image using poisson image blending

    Args:
        img (np.array): background image
        smoke (np.array): smoke image
        offset (tuple, optional): smoke location offset (dy, dx). Defaults to (0, 0).

    Returns:
        np.array: result image
        np.array: result mask
    """

    smoke_mask = smoke[:, :, 0] > 50
    smoke_mask, smoke, offset_adj = create_mask(smoke_mask, img, smoke, offset=offset)

    result = poisson_blend(
        smoke_mask, smoke, img, method="normal", offset_adj=offset_adj
    )

    mask = img[:, :, 0] * 0
    dy, dx = offset
    mask[dy : dy + smoke_mask.shape[0], dx : dx + smoke_mask.shape[1]] = smoke_mask

    return result, mask
