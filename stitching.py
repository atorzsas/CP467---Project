import cv2
import numpy as np
import os

# Load the images
image_files = [r'Scene Images/S23.png', r'Scene Images/S22.png', r'Scene Images/S21.png']
images = [cv2.imread(p) for p in image_files]

# Check if images are loaded
for i, img in enumerate(images):
    if img is None:
        raise ValueError(f"Error loading image: {image_files[i]}")

# Convert to grayscale
gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

# Use SIFT for feature detection and extraction
sift = cv2.SIFT_create()

# Detect features and compute descriptors
keypoints_all = []
descriptors_all = []
for gray in gray_images:
    kp, des = sift.detectAndCompute(gray, None)
    keypoints_all.append(kp)
    descriptors_all.append(des)

# Match descriptors between pairs of images
bf = cv2.BFMatcher()
matches_all = []
for i in range(len(images) - 1):
    matches = bf.knnMatch(descriptors_all[i], descriptors_all[i+1], k=2)
    # Apply ratio test to find good matches
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    matches_all.append(good)

# Function to warp and combine two images
def stitch_images(img1, img2, kp1, kp2, matches):
    # Extract location of good matches
    points1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find homography
    H, _ = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography to warp images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img1_warped = cv2.warpPerspective(img1, H, (w1+w2, h1))
    # Replace the target area with the source image
    img1_warped[0:h2, 0:w2] = img2
    return img1_warped

# Initialize final image as the first image
final_img = images[0]

# Sequentially stitch images
for i in range(len(images) - 1):
    final_img = stitch_images(final_img, images[i+1], keypoints_all[i], keypoints_all[i+1], matches_all[i])


# Save the stitched image
output_file = 'Stitched_Scene.jpg'
cv2.imwrite(output_file, final_img)

print(f"Stitched panorama saved as {output_file}")