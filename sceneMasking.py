"""
Used to mask the scene images before you find the keypoints
"""

import cv2
import numpy as np


originalScene = 'Scene Images/'
maskedScenes = 'Masked Scene Images/'


for i in range(1, 30):
    image_path = f'{originalScene}S{i}.png'
    image = cv2.imread(image_path)

    print(f"Masking Scene {i}")

    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_brown = np.array([5, 100, 58])
    upper_brown = np.array([163, 235, 198])

   
    mask = cv2.inRange(hsv, lower_brown, upper_brown)

    # Invert the mask to get the objects
    mask = cv2.bitwise_not(mask)

    # Perform opening to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Apply the mask to the image
    result = cv2.bitwise_and(image, image, mask=mask)


    output_path = f'{maskedScenes}S{i}_masked.png'
    cv2.imwrite(output_path, result)