# Copyright (C) 2019-2022, Pyronear.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import cv2
import numpy as np
import os


def read_video(file, size_max=1280):
    # Read smoke video
    cap = cv2.VideoCapture(file)
    imgs = []
    ret = True
    while cap.isOpened() and ret:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            r = size_max / max(frame.shape)
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (int(w * r), int(h * r)))
            imgs.append(frame)

    return imgs


def save_img(folder_path, filename, img):
    os.makedirs(folder_path, exist_ok=True)
    cv2.imwrite(f"{folder_path}/{filename}", img)


def save_label(folder_path, filename, label):
    os.makedirs(folder_path, exist_ok=True)
    with open(f"{folder_path}/{filename}", "w") as f:
        f.write(label)


def get_label(mask):
    """Compute bounding box"""
    if np.sum(mask > 0) > 100:  # set minimum size (10x10 pixel on average)
        y0, x0 = np.min(np.where(mask), 1)
        y1, x1 = np.max(np.where(mask), 1)
        h, w = mask.shape
        dw = (x1 - x0) / w
        dh = (y1 - y0) / h
        xc = x0 / w + dw / 2
        yc = y0 / h + dh / 2
        label = str(1) + " " + str(xc) + " " + str(yc) + " " + str(dw) + " " + str(dh)
    else:
        label = ""

    return label
