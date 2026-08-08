"""
Noise Quality Assessment

This module estimates image noise using
the standard deviation of the high-frequency residual.
"""

import cv2
import numpy as np


def calculate_noise_score(image):
    """
    Calculate image noise score.

    Parameters
    ----------
    image : numpy.ndarray

    Returns
    -------
    float
    """

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    residual = gray.astype(np.float32) - blurred.astype(np.float32)

    noise = np.std(residual)

    return float(noise)