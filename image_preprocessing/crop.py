import numpy as np
import cv2
from PIL import Image


def get_cropping_coordinates(image, padding=100):

    ### uses contour finding algorithm from cv2 to remove black spaces from images
    ### image: 8bit pixel array

    _, thresh = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)

    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1], w + 2 * padding)
    h = min(image.shape[0], h + 2 * padding)

    return x, y , w, h


def crop_image(image, box_coordinates):

    ### image: 8bit or 16bit pixel array

    x, y, w, h = box_coordinates
    return image[y:y+h, x:x+w]

def pad_crop_image(image, box_coordinates):

    ### Due to padding and perfect square bounding box cropping, top left and/or bottom right corners might not lie on the image
    ### This code pads the image with black pixel values in that case
    ### image: 8bit or 16 bit pixel array

    x, y, w, h = box_coordinates
    H, W = image.shape

    paddingX1 = int(max(-1*x, 0))
    paddingY1 = int(max(-1*y, 0))
    paddingX2 = int(max(0, x + w - W))
    paddingY2 = int(max(0, y + h - H))

    x = x + paddingX1
    y = y + paddingY1

    padded_image = np.pad(image, pad_width=((paddingY1, paddingY2), (paddingX1, paddingX2)), mode='constant', constant_values=0)
    padded_crop = padded_image[y:y+h, x:x+w]

    return padded_crop