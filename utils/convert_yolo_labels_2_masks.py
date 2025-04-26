import os

import cv2
import numpy as np
from tqdm import tqdm


def scale_polygon(points, scale_factor=0.95):
    centroid = np.mean(points, axis=0)
    scaled_points = (points - centroid) * scale_factor + centroid
    return scaled_points.astype(np.int32)


def create_mask_from_yolo(image_path, label_path, output_mask_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    height, width = img.shape[:2]
    sep_mask = np.zeros((height, width), dtype=np.uint8)

    with open(label_path, "r") as f:
        lines = f.readlines()

    if not lines:
        print(f"Label file is empty: {label_path}")
        return

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            print(f"Invalid label format in {label_path}: {line}")
            continue
        class_id = int(parts[0])
        if class_id != 0:  # class 0 for rocks
            continue
        points = np.array([float(p) for p in parts[1:]]).reshape(-1, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        points = points.astype(np.int32)

        # Draw each instance separately on sep_mask
        scaled_points = scale_polygon(points, scale_factor=0.9)
        cv2.fillPoly(sep_mask, [scaled_points], 1)

    cv2.imwrite(output_mask_path, sep_mask * 255)


def process_folder(image_dir, label_dir, mask_dir):
    os.makedirs(mask_dir, exist_ok=True)

    for label_name in tqdm(os.listdir(label_dir), desc="Processing labels"):
        if not label_name.endswith(".txt"):
            continue

        label_path = os.path.join(label_dir, label_name)
        image_name = label_name.replace(".txt", ".jpg")
        image_path = os.path.join(image_dir, image_name)
        mask_name = label_name.replace(".txt", ".png")
        mask_path = os.path.join(mask_dir, mask_name)

        if not os.path.exists(image_path):
            print(f"Image not found for label {label_path}: {image_path}")
            continue

        create_mask_from_yolo(image_path, label_path, mask_path)


if __name__ == "__main__":
    base_dir = "data"

    for split in ["train", "valid", "test"]:
        image_dir = os.path.join(base_dir, split, "images")
        label_dir = os.path.join(base_dir, split, "labels")
        mask_dir = os.path.join(base_dir, split, "masks")

        print(f"Processing {split} set...")
        process_folder(image_dir, label_dir, mask_dir)
        print(f"Finished processing {split} set.")
