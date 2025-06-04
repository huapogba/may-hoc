import os

import cv2
import numpy as np


def center_and_resize(image, size=48):
    # Convert to grayscale and binary (invert if necessary)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours and the largest one (assuming that's the digit)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Get bounding rect of the largest contour
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    # Crop the digit
    digit = binary[y : y + h, x : x + w]

    # Make square canvas
    size_max = max(w, h)
    square = np.zeros((size_max, size_max), dtype=np.uint8)

    # Center the digit in the square canvas
    x_offset = (size_max - w) // 2
    y_offset = (size_max - h) // 2
    square[y_offset : y_offset + h, x_offset : x_offset + w] = digit

    # Resize to desired size
    final = cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)
    return final


def process_all_images(input_dir, output_dir="output_48x48"):
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                path = os.path.join(root, fname)
                img = cv2.imread(path)
                result = center_and_resize(img)
                if result is not None:
                    # Save with the same filename structure
                    rel_path = os.path.relpath(path, input_dir)
                    save_path = os.path.join(output_dir, rel_path)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    cv2.imwrite(save_path, result)
                    count += 1
    print(f"Processed {count} images.")


if __name__ == "__main__":
    process_all_images(".")  # "." is the current directory
