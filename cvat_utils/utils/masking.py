import cv2
import numpy as np
from PIL import Image, ImageDraw

def get_foreground_background(image, points):
    height, width, _ = image.shape
    img = Image.new('L', (int(width), int(height)), 0)
    ImageDraw.Draw(img).polygon(points, outline=1, fill=1)
    mask = np.array(img)

    foreground_mask = np.logical_not(mask).astype('uint8')

    result_foreground = cv2.bitwise_and(image, image, mask=mask)
    result_backround = cv2.bitwise_and(image, image, mask=foreground_mask)

    return result_foreground, result_backround