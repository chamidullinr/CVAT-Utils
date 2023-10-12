from typing import Tuple

import cv2
import numpy as np


def get_foreground_background(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Takes image and mask and generates two images, one without bgr and one without frg.

    Parameters
    ----------
    image
        Given image as np.ndarray
    mask:
        Binary mask as np.ndarray
    Returns
    -------
        Tuple with two images (i.e., np.arrays) with removed foreground and background.
    """
    foreground_mask = np.logical_not(mask).astype("uint8")
    masked_foreground = cv2.bitwise_and(image, image, mask=mask)
    masked_backround = cv2.bitwise_and(image, image, mask=foreground_mask)

    return masked_foreground, masked_backround
