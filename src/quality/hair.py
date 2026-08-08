"""
Hair Density Assessment

This module estimates the percentage
of hair pixels using Black-Hat morphology.
"""

import cv2
import numpy as np


def calculate_hair_density(image):
    """
    Calculate hair density.

    Parameters
    ----------
    image : numpy.ndarray

    Returns
    -------
    float
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (17, 17)
    )

    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    _, mask = cv2.threshold(
        blackhat,
        15,
        255,
        cv2.THRESH_BINARY
    )

    hair_pixels = np.count_nonzero(mask)

    total_pixels = mask.size

    density = hair_pixels / total_pixels

    return float(density)