"""
Brightness Quality Assessment

This module computes the average brightness
of a dermoscopic image.
"""

import cv2
import numpy as np


def calculate_brightness_score(image):
    """
    Calculate average image brightness.

    Parameters
    ----------
    image : numpy.ndarray
        RGB image.

    Returns
    -------
    float
        Brightness score.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    brightness = np.mean(gray)

    return float(brightness)