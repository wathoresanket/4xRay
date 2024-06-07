import os
import sys

# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate one folder back from the current directory
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# Append the parent directory to sys.path
sys.path.append(parent_dir)

import json
import random
import shutil
import numpy as np
from PIL import Image, ImageDraw


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def process_seg_quadrant():
    """
    Generate segmentation masks for quadrants.
    """
    dataset_json = load_json("dentex_dataset/origin/quadrant/train_quadrant.json")
    quadrant_remap = {
        0: 1,
        1: 0,
        2: 2,
        3: 3,
    }  
    mkdirs("dentex_dataset/segmentation/quadrant/masks")
    mkdirs("dentex_dataset/segmentation/quadrant/xrays")

    image_names = []
    for image_info in dataset_json["images"]:
        image_names.append(image_info["file_name"])
        # draw mask for each image
        image = Image.open(f"dentex_dataset/origin/quadrant/xrays/{image_info['file_name']}")
        mask = Image.new("L", image.size)
        draw = ImageDraw.Draw(mask)

        for annotation in dataset_json["annotations"]:
            if annotation["image_id"] == image_info["id"]:
                points = np.array(annotation["segmentation"]).reshape(-1, 2)
                points = [tuple(point) for point in points]
                # Get the remapped category ID
                category_id = quadrant_remap[annotation["category_id"]]
                # Fill the polygon with the remapped category ID
                draw.polygon(points, fill=category_id + 1)

        # save mask and copy image
        mask.save(f"dentex_dataset/segmentation/quadrant/masks/{image_info['file_name']}")
        shutil.copy(
            f"dentex_dataset/origin/quadrant/xrays/{image_info['file_name']}",
            f"dentex_dataset/segmentation/quadrant/xrays/{image_info['file_name']}",
        )

    save_json("dentex_dataset/segmentation/quadrant/image_names.json", image_names)
    
if __name__ == "__main__":
    process_seg_quadrant()
    ...