import cv2
import os
import numpy as np

# Base project path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input folders
input_folders = {
    "helmet": os.path.join(BASE_DIR, "dataset", "helmet"),
    "no_helmet": os.path.join(BASE_DIR, "dataset", "no_helmet")
}

# Output folders
output_folders = {
    "helmet": os.path.join(BASE_DIR, "dataset", "processed_helmet"),
    "no_helmet": os.path.join(BASE_DIR, "dataset", "processed_no_helmet")
}

# Create output folders
for folder in output_folders.values():
    os.makedirs(folder, exist_ok=True)

# CLAHE enhancement
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocess_image(image_path):

    img = cv2.imread(image_path)

    if img is None:
        print(f"Skipped unreadable file: {image_path}")
        return None

    # Resize
    img = cv2.resize(img, (224, 224))

    # Sharpening kernel
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    # Apply sharpening
    img = cv2.filter2D(img, -1, kernel)

    # Enhance contrast
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))

    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return img

# Process images
for label in input_folders:

    input_folder = input_folders[label]
    output_folder = output_folders[label]

    print(f"Processing {label} images from: {input_folder}")

    if not os.path.exists(input_folder):
        print(f"Folder not found: {input_folder}")
        continue

    for file_name in os.listdir(input_folder):

        input_path = os.path.join(input_folder, file_name)

        processed_img = preprocess_image(input_path)

        if processed_img is not None:

            output_path = os.path.join(output_folder, file_name)

            cv2.imwrite(output_path, processed_img)

print("Preprocessing completed successfully.")