"""
Contrast Quality Assessment

This module computes the global contrast score of a dermoscopic image
using the standard deviation of grayscale intensity values.
"""

import cv2
import numpy as np


def calculate_contrast_score(image):
    """
    Calculate the contrast score of an RGB image.

    Parameters
    ----------
    image : numpy.ndarray
        RGB image.

    Returns
    -------
    float
        Contrast score.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    contrast = np.std(gray)

    return float(contrast)