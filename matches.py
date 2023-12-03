import cv2
import numpy as np
import os
import re

# Define a function to extract numbers from a filename
def extract_numbers(filename):
    return re.findall(r'\d+', filename)

# Load the scene and object images
scene_image_path = r'Scene Images/S20.png'  # Change Scene Accordinly
object_image_path = r'Isolated Images/O11.png'  # Change Object Accordingly

scene_image = cv2.imread(scene_image_path)
object_image = cv2.imread(object_image_path)

# Check if the images are loaded correctly
if scene_image is None or object_image is None:
    raise ValueError("Could not load one or more images.")

# Extract scene and object numbers from filenames
scene_number = extract_numbers(os.path.basename(scene_image_path))[0]
object_number = extract_numbers(os.path.basename(object_image_path))[0]

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Find the keypoints and descriptors with SIFT
kp_scene, des_scene = sift.detectAndCompute(scene_image, None)
kp_object, des_object = sift.detectAndCompute(object_image, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

# FLANN based matcher
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des_object, des_scene, k=2)

# Filter out weak matches with Lowe's ratio test
good = []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# Draw only good matches
matched_image = cv2.drawMatchesKnn(object_image, kp_object, scene_image, kp_scene, good, None, flags=2)

# Create the Matches folder if it doesn't exist
matches_folder = 'Matches'
os.makedirs(matches_folder, exist_ok=True)

# Save the matched image with dynamic naming based on the scene and object numbers
matched_image_filename = f'S{scene_number}_O{object_number}_matches.jpg'
matched_image_path = os.path.join(matches_folder, matched_image_filename)
cv2.imwrite(matched_image_path, matched_image)

print(f"Matched image saved at {matched_image_path}")
