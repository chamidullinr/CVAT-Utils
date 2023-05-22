import numpy as np


def rle_to_mask(points: list, image_height: int, image_width: int) -> np.ndarray:
    """Decode data compressed with CVAT-based Run-Length Encoding (RLE) and return image mask.

    Parameters
    ----------
    points
        List of points stored in shape with mask type returned by CVAT API.
    image_height
        Height of image.
    image_width
        Width of image.

    Returns
    -------
    mask
        A 2D NumPy array that represent image mask.
    """
    points = np.array(points).astype(int)
    rle = points[:-4]
    left, top, right, bottom = points[-4:]
    width = right - left + 1
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    value, offset = 0, 0
    for rle_count in rle:
        while rle_count > 0:
            x, y = offset % width, offset // width
            mask[y + top][x + left] = value
            rle_count -= 1
            offset += 1
        value = abs(value - 1)
    return mask


def mask_to_points(mask: np.ndarray) -> list:
    """Convert 2D mask into a 1D flattened list of points.

    Parameters
    ----------
    mask
        A 2D NumPy array that represent image mask.

    Returns
    -------
    points
        A 1D flattened list of points.
    """
    return np.array(np.where(mask)[::-1]).T.reshape(-1).tolist()
