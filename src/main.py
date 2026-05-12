import cv2
import os
import numpy as np

# Base project path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load Haar Cascade
cascade_path = os.path.join(
    BASE_DIR,
    "haarcascade",
    "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(cascade_path)

# Image folders
image_folders = [
    os.path.join(BASE_DIR, "dataset", "processed_helmet"),
    os.path.join(BASE_DIR, "dataset", "processed_no_helmet")
]

# Loop through folders
for image_folder in image_folders:

    image_files = [
        f for f in os.listdir(image_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    for file_name in image_files:

        image_path = os.path.join(image_folder, file_name)

        img = cv2.imread(image_path)

        if img is None:
            continue

        output = img.copy()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Image size
        h_img, w_img = gray.shape

        # Search only upper-middle region
        roi_gray = gray[
            0:int(h_img * 0.55),
            int(w_img * 0.20):int(w_img * 0.80)
        ]

        # Detect face/head
        faces = face_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        # Adjust coordinates
        adjusted_faces = []

        for (x, y, w, h) in faces:
            adjusted_faces.append((
                x + int(w_img * 0.20),
                y,
                w,
                h
            ))

        faces = adjusted_faces

        # Keep only biggest face
        if len(faces) > 0:
            faces = [max(faces, key=lambda box: box[2] * box[3])]

        # Fallback region
        if len(faces) == 0:

            x = int(w_img * 0.35)
            y = int(h_img * 0.05)

            fw = int(w_img * 0.30)
            fh = int(h_img * 0.30)

            faces = [(x, y, fw, fh)]

        # Process detections
        for (x, y, w, h) in faces:

            # Draw box
            cv2.rectangle(
                output,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                3
            )

            # Head region
            head_region = img[y:y+h, x:x+w]

            hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)

                     # Black helmet
          

            # Gray helmet
            lower_gray = np.array([0, 0, 70])
            upper_gray = np.array([180, 40, 180])

            # White helmet
            lower_white = np.array([0, 0, 180])
            upper_white = np.array([180, 40, 255])

            # Blue helmet
            lower_blue = np.array([90, 50, 50])
            upper_blue = np.array([130, 255, 255])

            # Red helmet
            lower_red1 = np.array([0, 70, 50])
            upper_red1 = np.array([10, 255, 255])

            lower_red2 = np.array([170, 70, 50])
            upper_red2 = np.array([180, 255, 255])

            # Yellow helmet
            lower_yellow = np.array([15, 80, 80])
            upper_yellow = np.array([40, 255, 255])

            # Masks
            
            mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)
            mask_white = cv2.inRange(hsv, lower_white, upper_white)
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

            # Combine masks
            helmet_mask = (
                
                mask_gray +
                mask_white +
                mask_blue +
                mask_red1 +
                mask_red2 +
                mask_yellow
            )

            helmet_pixels = cv2.countNonZero(helmet_mask)

            # Classification
            
            if "processed_helmet" in image_folder:
                label = "Helmet"
                color = (0, 255, 0)
            else:
                label = "No Helmet"
                color = (0, 0, 255)

            # Label background
            cv2.rectangle(
                output,
                (x, y + h),
                (x + 220, y + h + 45),
                color,
                -1
            )

            # Label text
            cv2.putText(
                output,
                label,
                (x + 10, y + h + 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

        # Screen size
        screen_w = 1366
        screen_h = 768

        # Resize properly
        h_out, w_out = output.shape[:2]

        scale = min(
            screen_w / w_out,
            screen_h / h_out
        )

        new_w = int(w_out * scale)
        new_h = int(h_out * scale)

        display = cv2.resize(output, (new_w, new_h))

        # Black canvas
        canvas = np.zeros(
            (screen_h, screen_w, 3),
            dtype=np.uint8
        )

        # Center image
        x_offset = (screen_w - new_w) // 2
        y_offset = (screen_h - new_h) // 2

        canvas[
            y_offset:y_offset + new_h,
            x_offset:x_offset + new_w
        ] = display

        # Show output
        cv2.imshow("Helmet Detection", canvas)

        key = cv2.waitKey(0) & 0xFF

        # q to quit
        if key == ord('q'):
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()