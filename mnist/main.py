import os

import cv2
import numpy as np
from PIL import Image
from rembg import remove
from tqdm import tqdm


def remove_background_and_save(input_path, output_path):
    try:
        input = Image.open(input_path)
        result = remove(input)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result.save(output_path)
    except Exception as e:
        print(f"[ERROR] {input_path}: {e}")


def process_images_remove_bg(input_dir, output_dir="output_nobg"):
    os.makedirs(output_dir, exist_ok=True)
    supported_exts = (".png", ".jpg", ".jpeg", ".bmp")
    count = 0

    for root, _, files in os.walk(input_dir):
        for fname in files:
            if fname.lower().endswith(supported_exts):
                input_path = os.path.join(root, fname)
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(
                    output_dir, os.path.splitext(rel_path)[0] + ".png"
                )
                remove_background_and_save(input_path, output_path)
                count += 1

    print(f"✅ Done. {count} images processed and saved to '{output_dir}'.")


if __name__ == "__main__":
    process_images_remove_bg(".")
