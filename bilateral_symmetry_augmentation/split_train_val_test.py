import os
import sys

# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate one folder back from the current directory
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# Append the parent directory to sys.path
sys.path.append(parent_dir)

import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as F
from models.unet.utils import load_seunet
import matplotlib.pyplot as plt
import json
import shutil
import random

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def process_seg_enumeration32_train_val_test():
    """
    draw segmentation masks for enumeration32
    """
    dataset_json = load_json("dentex_dataset/origin/quadrant_enumeration/train_quadrant_enumeration.json")
    mkdirs("dentex_dataset/segmentation/enumeration32_train_val_test/train/masks")
    mkdirs("dentex_dataset/segmentation/enumeration32_train_val_test/train/xrays")
    mkdirs("dentex_dataset/segmentation/enumeration32_train_val_test/val/masks")
    mkdirs("dentex_dataset/segmentation/enumeration32_train_val_test/val/xrays")
    mkdirs("dentex_dataset/segmentation/enumeration32_train_val_test/test/masks")
    mkdirs("dentex_dataset/segmentation/enumeration32_train_val_test/test/xrays")

    all_image_names = [image_info["file_name"] for image_info in dataset_json["images"]]
    selected_train_image_names = random.sample(all_image_names, 380)
    remaining_image_names = [image_name for image_name in all_image_names if image_name not in selected_train_image_names]
    selected_val_image_names = random.sample(remaining_image_names, 127)
    remaining_image_names = [image_name for image_name in remaining_image_names if image_name not in selected_val_image_names]

    train_image_names = []
    for image_name in selected_train_image_names:
        train_image_names.append(image_name)
        # draw mask for each image
        image_info = next((image_info for image_info in dataset_json["images"] if image_info["file_name"] == image_name), None)
        if image_info:
            image = Image.open(f"dentex_dataset/origin/quadrant_enumeration/xrays/{image_info['file_name']}")
            mask = Image.new("L", image.size)
            draw = ImageDraw.Draw(mask)

            for annotation in dataset_json["annotations"]:
                if annotation["image_id"] == image_info["id"]:
                    points = np.array(annotation["segmentation"]).reshape(-1, 2)
                    points = [tuple(point) for point in points]
                    # draw polygon, fill with label 1~32
                    draw.polygon(points, fill=annotation["category_id_1"] * 8 + annotation["category_id_2"] + 1)

            # save mask and copy image to train folder
            mask.save(f"dentex_dataset/segmentation/enumeration32_train_val_test/train/masks/{image_info['file_name']}")
            shutil.copy(
                f"dentex_dataset/origin/quadrant_enumeration/xrays/{image_info['file_name']}",
                f"dentex_dataset/segmentation/enumeration32_train_val_test/train/xrays/{image_info['file_name']}",
            )

    val_image_names = []
    for image_name in selected_val_image_names:
        val_image_names.append(image_name)
        # draw mask for each image
        image_info = next((image_info for image_info in dataset_json["images"] if image_info["file_name"] == image_name), None)
        if image_info:
            image = Image.open(f"dentex_dataset/origin/quadrant_enumeration/xrays/{image_info['file_name']}")
            mask = Image.new("L", image.size)
            draw = ImageDraw.Draw(mask)

            for annotation in dataset_json["annotations"]:
                if annotation["image_id"] == image_info["id"]:
                    points = np.array(annotation["segmentation"]).reshape(-1, 2)
                    points = [tuple(point) for point in points]
                    # draw polygon, fill with label 1~32
                    draw.polygon(points, fill=annotation["category_id_1"] * 8 + annotation["category_id_2"] + 1)

            # save mask and copy image to val folder
            mask.save(f"dentex_dataset/segmentation/enumeration32_train_val_test/val/masks/{image_info['file_name']}")
            shutil.copy(
                f"dentex_dataset/origin/quadrant_enumeration/xrays/{image_info['file_name']}",
                f"dentex_dataset/segmentation/enumeration32_train_val_test/val/xrays/{image_info['file_name']}",
            )

    test_image_names = []
    for image_name in remaining_image_names:
        test_image_names.append(image_name)
        # draw mask for each image
        image_info = next((image_info for image_info in dataset_json["images"] if image_info["file_name"] == image_name), None)
        if image_info:
            image = Image.open(f"dentex_dataset/origin/quadrant_enumeration/xrays/{image_info['file_name']}")
            mask = Image.new("L", image.size)
            draw = ImageDraw.Draw(mask)

            for annotation in dataset_json["annotations"]:
                if annotation["image_id"] == image_info["id"]:
                    points = np.array(annotation["segmentation"]).reshape(-1, 2)
                    points = [tuple(point) for point in points]
                    # draw polygon, fill with label 1~32
                    draw.polygon(points, fill=annotation["category_id_1"] * 8 + annotation["category_id_2"] + 1)

            # save mask and copy image to test folder
            mask.save(f"dentex_dataset/segmentation/enumeration32_train_val_test/test/masks/{image_info['file_name']}")
            shutil.copy(
                f"dentex_dataset/origin/quadrant_enumeration/xrays/{image_info['file_name']}",
                f"dentex_dataset/segmentation/enumeration32_train_val_test/test/xrays/{image_info['file_name']}",
            )

    save_json("dentex_dataset/segmentation/enumeration32_train_val_test/train/image_names.json", train_image_names)
    save_json("dentex_dataset/segmentation/enumeration32_train_val_test/val/image_names.json", val_image_names)
    save_json("dentex_dataset/segmentation/enumeration32_train_val_test/test/image_names.json", test_image_names)


if __name__ == "__main__":
    process_seg_enumeration32_train_val_test()
    ...
