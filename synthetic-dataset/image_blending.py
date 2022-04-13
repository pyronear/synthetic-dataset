import cv2
import numpy as np


def basic_blending(img, smoke, offset=(0, 0), opacity=0.8):
    """Add smoke on image using basic image blending

    Args:
        img (np.array): background image
        smoke (np.array): smoke image
        offset (tuple, optional): smoke location offset (dy, dx). Defaults to (0, 0).
        opacity (float, optional): smoke image opacity in [0, 1]. Defaults to (0, 0).

    Returns:
        _type_: _description_
    """
    ks = 7
    kernel = np.ones((ks, ks), np.float32)/(ks**2)

    dy, dx = offset
    temp = img[dy:dy+smoke.shape[0], dx:dx+smoke.shape[1], :]
    dst = cv2.filter2D(smoke, -1, kernel)
    mask_dst = dst > 50
    alpha = 1 - opacity * dst/np.max(dst)
    res = temp*alpha + smoke[:, :, ::-1]*(1-alpha)
    img[dy:dy+smoke.shape[0], dx:dx+smoke.shape[1], :] = res
    mask = img * 0
    mask[dy:dy+smoke.shape[0], dx:dx+smoke.shape[1], :] = mask_dst

    return img, mask
