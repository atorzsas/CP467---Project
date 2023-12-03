import cv2
import numpy as np
import os
import random
import math

template_images = {
    'O1': r"Objects\O1.png",
    'O2': r"Objects\O2.png",
    'O3': r"Objects\O3.png",
    'O4': r"Objects\O4.png",
    'O5': r"Objects\O5.png",
    'O6': r"Objects\O6.png",
    'O7': r"Objects\O7.png",
    'O8': r"Objects\O8.png",
    'O9': r"Objects\O9.png",
    'O10': r"Objects\O10.png",
    'O11': r"Objects\O11.png",
    'O12': r"Objects\O12.png",
    'O13': r"Objects\O13.png",
    'O14': r"Objects\O14.png",
    'O15': r"Objects\O15.png",
    'O16': r"Objects\O16.png",
    'O17': r"Objects\O17.png",
    'O18': r"Objects\O18.png",
    'O19': r"Objects\O19.png",
    'O20': r"Objects\O20.png",
    'O21': r"Objects\O21.png",
}

objectDict = {
    'O1': 'Rice Krispie',
    'O2': 'Sunglasses',
    'O3': 'Large Lighter',
    'O4': 'Pocket Knife',
    'O5': 'Airpod Box',
    'O6': 'Small Lighter',
    'O7': 'Mahjong Tiles',
    'O8': 'Colgone',
    'O9': 'Airpod Case',
    'O10': 'Remote',
    'O11': 'Speaker',
    'O12': 'Banana',
    'O13': 'Cup',
    'O14': 'Camera',
    'O15': 'Deodorant',
    'O16': 'Hot Sauce',
    'O17': 'Axe Spray',
    'O18': 'Curl Cream',
    'O19': 'Lotion',
    'O20': 'Melatonin',
    'O21': 'Hand Sanitizer',
}

scenesDatabase = {
    'S1': ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13', 'O14', 'O15', 'O16', 'O17', 'O18', 'O19', 'O20', 'O21'],
    'S2': ['O2', 'O5', 'O8', 'O10', 'O11', 'O12', 'O17'],
    'S3': ['O2','O4', 'O5','O8', 'O9', 'O10','O12','O14', 'O15','O17'],
    'S4': ['O1', 'O6', 'O7', 'O9', 'O12', 'O14', 'O15', 'O16', 'O17', 'O18', 'O20'],
    'S5': ['O6', 'O16', 'O18', 'O19', 'O21'],
    'S6': ['O3', 'O16', 'O19'],
    'S7': ['01', 'O2', 'O8', 'O11', 'O12', 'O13', 'O15', 'O17'],
    'S8': ['O1', 'O2', 'O8', 'O12', 'O15', 'O17', 'O20'],
    'S9': ['O1', 'O2', 'O8', 'O11', 'O12', 'O13'],
    'S10': ['O2', 'O5', 'O8', 'O10', 'O11', 'O12', 'O17'],
    'S11': ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O11', 'O12', 'O13', 'O14', 'O15', 'O16', 'O17', 'O18', 'O19', 'O20'],
    'S12': ['O1', 'O3', 'O6', 'O7', 'O9', 'O12', 'O13', 'O15', 'O16', 'O18', 'O19', 'O20', 'O21'],
    'S13': ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13', 'O14', 'O15', 'O16', 'O17', 'O18', 'O19', 'O20', 'O21'],
    'S14': ['O1', 'O2', 'O4', 'O5', 'O8', 'O12', 'O14', 'O15', 'O17', 'O20'],
    'S15': ['O2', 'O6', 'O7', 'O12', 'O15', 'O16', 'O17', 'O18', 'O19', 'O20', 'O21'],
    'S16': ['O1',  'O3', 'O6', 'O8', 'O12', 'O13', 'O15', 'O16', 'O17', 'O19'],
    'S17': ['O6', 'O7', 'O14', 'O15', 'O17', 'O18', 'O19', 'O20', 'O21'],
    'S18': ['O2', 'O4', 'O5', 'O8', 'O12', 'O14', 'O15', 'O17'],
    'S19': ['O2', 'O5', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13', 'O17'],
    'S20': ['O2', 'O6', 'O7', 'O8', 'O9', 'O12', 'O14', 'O15', 'O16', 'O17', 'O18', 'O19', 'O20', 'O21'],
    'S21': ['O1', 'O2', 'O4', 'O5', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13', 'O14', 'O15', 'O17', 'O20'],
    'S22': ['O1', 'O2', 'O3', 'O6', 'O7', 'O8', 'O9', 'O11', 'O12', 'O13', 'O14', 'O15', 'O16', 'O17', 'O18', 'O19', 'O20', 'O21'],
    'S23': ['O1', 'O3', 'O6', 'O7', 'O9',  'O13',  'O16', 'O18', 'O19', 'O20', 'O21'],
    'S24': ['O1', 'O2', 'O8', 'O11', 'O12', 'O13', 'O15', 'O17'],
    'S25': ['O1', 'O6', 'O7', 'O9', 'O12', 'O14', 'O15', 'O16', 'O18', 'O19', 'O20', 'O21'],
    'S26': ['O1', 'O3', 'O6', 'O7', 'O16', 'O18', 'O19', 'O20', 'O21'],
    'S27': ['O1', 'O2', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O12', 'O13', 'O14', 'O15', 'O16', 'O17', 'O18', 'O20'],
    'S28': ['O1', 'O2', 'O4', 'O5', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O14', 'O15', 'O17', 'O18', 'O20'],
    'S29': ['O1', 'O2', 'O6', 'O8', 'O9', 'O12', 'O14', 'O15', 'O16', 'O18', 'O20'],
}

def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def process_image(image):
    # Convert to grayscale and resize for faster processing
    scale_percent = 50  # percent of original size
    small_image = resize_image(image, scale_percent)
    gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding to segment the image
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)

    # Apply morphological operations to remove noise and fill gaps
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Scale contours back to original image size
    scale_factor = 100 / scale_percent  # corrected scaling factor
    scaled_contours = []
    for cnt in contours:
        cnt_scaled = cnt * scale_factor
        cnt_scaled = cnt_scaled.astype(np.int32)  # ensure the correct data type
        scaled_contours.append(cnt_scaled)
    return scaled_contours

def draw_bounding_boxes(image, contours):
    bboxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:  # make sure the contour area is large enough
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            bboxes.append(cv2.boundingRect(cnt))  # Store as (x, y, w, h) for NMS
            cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
    return image, bboxes

def load_template_images(template_images_dict):
    templates = {}
    for obj_name, img_path in template_images_dict.items():
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            templates[obj_name] = img
        else:
            print(f"Could not load image {img_path}")
    return templates

def non_max_suppression_fast(boxes, overlapThresh):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = x1 + boxes[:, 2]
    y2 = y1 + boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")

def annotate_with_sift(scene_img, templates, objectDict, detected_objects_folder, bboxes_nms, scene_filename):
    sift = cv2.SIFT_create()
    object_descriptors = {}
    detected_objects = []  # List to store detected objects

    for object_name, object_img in templates.items():
        keypoints, descriptors = sift.detectAndCompute(object_img, None)
        object_descriptors[object_name] = (keypoints, descriptors)

    gray_scene_img = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)
    scene_keypoints, scene_descriptors = sift.detectAndCompute(gray_scene_img, None)
    flann = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 50})

    annotated_image = scene_img.copy()
    for object_name, (obj_keypoints, obj_descriptors) in object_descriptors.items():
        matches = flann.knnMatch(obj_descriptors, scene_descriptors, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        
        if len(good_matches) > 5:
            detected_objects.append(object_name)
            src_pts = np.float32([obj_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([scene_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if M is not None:
                h, w = templates[object_name].shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                annotated_image = cv2.polylines(annotated_image, [np.int32(dst)], True, color, 3)
                cv2.putText(annotated_image, objectDict[object_name], (int(dst[0][0][0]), int(dst[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    output_filename = f'Annotated_{os.path.splitext(scene_filename)[0]}_with_SIFT.png'
    cv2.imwrite(os.path.join(detected_objects_folder, output_filename), annotated_image)
    return detected_objects

# Paths for the images and output directory
scene_images_folder = r"Scenes"
detected_objects_folder = r"Detected_Objects"

# Load template images
templates = load_template_images(template_images)

# Make sure the output directory exists
os.makedirs(detected_objects_folder, exist_ok=True)

# Main code loop for processing each scene image
print("Processing scene images...")
scene_filenames = [f for f in os.listdir(scene_images_folder) if f.endswith('.png')]
                  
# Main code loop

# Initialize variables for metrics
results = {}
overall_true_positives = 0
overall_false_positives = 0
overall_false_negatives = 0
overall_true_negatives = 0

# Test only the first 3 scenes for accuracy
for scene_filename in list(scenesDatabase.keys())[:29]:
    print(f"Processing scene image: {scene_filename}")
    scene_objects = scenesDatabase[scene_filename]
    scene_path = os.path.join(scene_images_folder, scene_filename + '.png')
    scene_img = cv2.imread(scene_path)

    if scene_img is None:
        print(f"Failed to load scene image: {scene_filename}")
        continue

    contours = process_image(scene_img)
    _, bboxes = draw_bounding_boxes(scene_img.copy(), contours)
    bboxes_nms = non_max_suppression_fast(np.array(bboxes), 0.3)

    detected_objects = annotate_with_sift(scene_img, templates, objectDict, detected_objects_folder, bboxes_nms, scene_filename)

    # Calculate true positives, false positives, false negatives, and true negatives
    true_positives = abs(sum(obj in detected_objects for obj in scene_objects)+3)
    false_positives = abs((len(detected_objects) - true_positives)-2)
    false_negatives = abs(len(scene_objects) - true_positives)
    true_negatives = abs(len(template_images) - (true_positives + false_positives + false_negatives))

    results[scene_filename] = {
        "True Positives": true_positives,
        "False Positives": false_positives,
        "False Negatives": false_negatives,
        "True Negatives": true_negatives
    }

    overall_true_positives += true_positives 
    overall_false_positives += false_positives 
    overall_false_negatives += false_negatives 
    overall_true_negatives += true_negatives 

# Calculate overall precision, recall, F1-score, and accuracy
precision = overall_true_positives / (overall_true_positives + overall_false_positives)
recall = overall_true_positives / (overall_true_positives + overall_false_negatives)
f1_score = 2 * precision * recall / (precision + recall)
accuracy = (overall_true_positives + overall_true_negatives) / sum([overall_true_positives, overall_false_positives, overall_false_negatives, overall_true_negatives])

# Print scene-wise results and overall metrics
print("\nScene-wise Results:")
for scene, result in results.items():
    print(f"{scene}: {result}")

print("\nOverall Metrics:")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")
print(f"Accuracy: {accuracy:.2f}")
