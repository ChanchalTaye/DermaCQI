"""
Blur Quality Assessment

This module estimates image sharpness
using the Variance of Laplacian method.
"""

import cv2


def calculate_blur_score(image):
    """
    Calculate blur score.

    Higher value = Sharper image.

    Parameters
    ----------
    image : numpy.ndarray

    Returns
    -------
    float
    """

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    blur = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    return float(blur)