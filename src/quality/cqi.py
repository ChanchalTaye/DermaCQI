"""
Composite Quality Index (CQI)

This module combines multiple quality metrics
into a single quality score.
"""

import numpy as np


def normalize(value, minimum, maximum):
    """
    Normalize a value to [0, 1].
    """

    if maximum == minimum:
        return 0.0

    return (value - minimum) / (maximum - minimum)


def calculate_cqi(
    contrast,
    brightness,
    blur,
    noise,
    hair
):
    """
    Calculate Composite Quality Index.

    Returns
    -------
    float
    """

    contrast = normalize(contrast, 0, 100)

    brightness = 1 - abs(brightness - 128) / 128

    brightness = np.clip(
        brightness,
        0,
        1
    )

    blur = normalize(
        blur,
        0,
        500
    )

    noise = 1 - normalize(
        noise,
        0,
        20
    )

    hair = 1 - hair

    cqi = (

        0.25 * contrast +

        0.20 * brightness +

        0.25 * blur +

        0.15 * noise +

        0.15 * hair

    )

    return float(cqi)