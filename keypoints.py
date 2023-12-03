'''
Used to find keypoints for objects and scenes
'''

import cv2
import numpy as np
import os


object_images_folder = 'Isolated Images/'
scene_images_folder = 'Masked Scene Images/'
keypoints_folder = 'Keypoints/'

# Objects Keypoints
for i in range(1, 22):
    image_path = f'{object_images_folder}O{i}.png'
    image = cv2.imread(image_path)

    print(f"Detecting Object {i}")
    
    # Preprocess the image to enhance features
    # Use CLAHE to improve the contrast locally
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe_img = clahe.apply(gray)

    # Initialize SIFT with adjusted parameters
    sift = cv2.SIFT_create(contrastThreshold=0.04, edgeThreshold=10, nOctaveLayers=3)

    # Detect keypoints using the adjusted SIFT detector on the enhanced image
    keypoints, descriptors = sift.detectAndCompute(clahe_img, None)

    # Draw keypoints
    img_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))

    output_path = f'{keypoints_folder}O{i}_keypoints.png'
    cv2.imwrite(output_path, img_keypoints)

# Scene Keypoints
for i in range(1, 30):
    image_path = f'{scene_images_folder}S{i}_masked.png'
    image = cv2.imread(image_path)

    print(f"Detecting Scene {i}")
    
    # Preprocess the image to enhance features
    # Use CLAHE to improve the contrast locally
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe_img = clahe.apply(gray)

    # Initialize SIFT with adjusted parameters
    sift = cv2.SIFT_create(contrastThreshold=0.04, edgeThreshold=10, nOctaveLayers=3)

    # Detect keypoints using the adjusted SIFT detector on the enhanced image
    keypoints, descriptors = sift.detectAndCompute(clahe_img, None)

    # Draw keypoints
    img_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))

    output_path = f'{keypoints_folder}S{i}_keypoints.png'
    cv2.imwrite(output_path, img_keypoints)

