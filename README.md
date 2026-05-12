# Helmet Detection Project

A computer vision application that detects whether a person is wearing a helmet using OpenCV and image processing techniques.

## Overview

This project uses Haar Cascade classifiers and color-based detection to identify helmets in images. It processes images through preprocessing steps and then analyzes the head region to detect the presence of helmets based on color characteristics.

## Project Structure

```
Helmet_Detection_Project/
├── dataset/
│   ├── helmet/                      # Original helmet images
│   ├── no_helmet/                   # Original non-helmet images
│   ├── processed_helmet/            # Preprocessed helmet images
│   └── processed_no_helmet/         # Preprocessed non-helmet images
├── haarcascade/
│   └── haarcascade_frontalface_default.xml  # Haar Cascade classifier
├── output/                          # Output results directory
├── src/
│   ├── main.py                      # Main helmet detection script
│   └── preprocess.py                # Image preprocessing script
└── README.md                        # This file
```

## Features

- **Face Detection**: Uses Haar Cascade classifier to detect head/face regions
- **Color-Based Detection**: Identifies helmets by analyzing colors in HSV color space
  - Gray helmets
  - White helmets
  - Blue helmets
  - Red helmets
  - Yellow helmets
- **Image Preprocessing**: 
  - Image resizing (224x224)
  - Gaussian blur for noise reduction
  - Sharpening filters
  - CLAHE (Contrast Limited Adaptive Histogram Equalization) enhancement
- **Visual Output**: Displays annotated images with detection results and labels

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy

## Installation

1. Clone or download this project
2. Install required dependencies:
```bash
pip install opencv-python numpy
```

## Usage

### 1. Preprocess Images

First, organize your images in the `dataset/helmet` and `dataset/no_helmet` directories, then run the preprocessing script:

```bash
python src/preprocess.py
```

This will:
- Read images from raw dataset folders
- Apply preprocessing (resize, blur, sharpen)
- Save processed images to `processed_helmet` and `processed_no_helmet` folders

### 2. Run Helmet Detection

After preprocessing, run the detection script:

```bash
python src/main.py
```

This will:
- Load preprocessed images
- Detect head regions using Haar Cascade
- Analyze color distribution in the head region
- Classify as "Helmet" or "No Helmet"
- Display annotated images with labels

**Controls:**
- Press 'q' to exit the detection loop
- Press any other key to continue to the next image

## How It Works

### Face Detection
- Uses OpenCV's Haar Cascade classifier to detect head/face regions
- Focuses on the upper-middle region of images (0-55% height, 20-80% width)
- Selects the largest detected face region

### Helmet Detection
- Extracts the top 35% of the detected head region (helmet area)
- Converts the region to HSV color space
- Creates masks for various helmet colors
- Counts pixels matching helmet colors
- Classifies as "Helmet" if pixel count > 5000, otherwise "No Helmet"

### Color Ranges (HSV)
- **Gray**: H: 0-180, S: 0-40, V: 70-180
- **White**: H: 0-180, S: 0-40, V: 180-255
- **Blue**: H: 90-130, S: 50-255, V: 50-255
- **Red**: H: 0-10 or 170-180, S: 70-255, V: 50-255
- **Yellow**: H: 15-40, S: 80-255, V: 80-255

## Output

- Annotated images showing:
  - Green bounding box around detected head
  - Colored label background (green for "Helmet", red for "No Helmet")
  - Classification text

## Customization

You can modify:
- **Image size**: Change `(224, 224)` in `preprocess.py`
- **Detection threshold**: Modify `helmet_pixels > 5000` in `main.py`
- **Color ranges**: Adjust HSV ranges in `main.py` to detect different helmet colors
- **Cascade classifier**: Use different `.xml` files for different detection scenarios
- **Screen resolution**: Modify `screen_w` and `screen_h` in `main.py`

## Troubleshooting

- **No faces detected**: Ensure images have clear head/face regions; adjust ROI coordinates if needed
- **False positives/negatives**: Fine-tune the pixel threshold (5000) or color ranges
- **Blurry detections**: Adjust preprocessing parameters (Gaussian blur, sharpening kernel)
- **Performance issues**: Reduce image size or process fewer images at a time

## Dependencies

| Package | Purpose |
|---------|---------|
| OpenCV | Image processing and face detection |
| NumPy | Array operations |

## Notes

- The Haar Cascade classifier works best on frontal face images
- Performance may vary based on image quality, lighting, and helmet types
- Color-based detection assumes helmets have distinct colors different from the head

## License

This project is provided as-is for educational and research purposes.

## Author

Created for helmet detection and safety compliance applications.
