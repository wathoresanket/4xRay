import os
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

def convert_mask_32_to_9(mask: np.ndarray, quadrant: int) -> np.ndarray:
    """
    convert mask from 32 classes to 9 classes,
    when a foreground label belongs to quadrant, it is converted to 1~8,
    when a foreground label does not belong to quadrant, it is converted to 9
    """
    assert quadrant in [0, 1, 2, 3]

    mask_out = mask.copy()
    mask_out[mask != 0] = 9

    for i in range(1, 9):
        mask_out[mask == (i + quadrant * 8)] = i

    return mask_out

def flip_quadrants_to_zero(image, mask, quadrant_id):
    if quadrant_id == 1:
        image = np.flip(image, axis=1)  # flip along y-axis
        mask = np.flip(mask, axis=1)  # flip along y-axis
    elif quadrant_id == 2:
        image = np.flip(image, axis=0)  # flip along x-axis
        image = np.flip(image, axis=1)  # flip along y-axis
        mask = np.flip(mask, axis=0)  # flip along x-axis
        mask = np.flip(mask, axis=1)  # flip along y-axis
    elif quadrant_id == 3:
        image = np.flip(image, axis=0)  # flip along x-axis
        mask = np.flip(mask, axis=0)  # flip along x-axis
    return image, mask

def process_seg_enumeration9_train_range(quadrant_prediction_path: str, size: int):
    quadrant_predictions = load_json(quadrant_prediction_path)
    quadrant_remap = {
        0: 1,
        1: 0,
        2: 2,
        3: 3,
    }
    # remap because category names are different between quadrant and quadrant_enumeration/quadrant_enumeration_disease

    output_xrays_dir = f"dentex_dataset/segmentation/enumeration9_quadrant_symmetry_augmentation/train_{size}/xrays/"
    output_masks_dir = f"dentex_dataset/segmentation/enumeration9_quadrant_symmetry_augmentation/train_{size}/masks/"
    mkdirs(output_xrays_dir)
    mkdirs(output_masks_dir)

    image_names = []
    for prediction_result in quadrant_predictions:
        file_name = prediction_result["file_name"]
        image_path = f"dentex_dataset/segmentation/enumeration32_train_val_test/train_{size}/xrays/{file_name}"
        mask_path = f"dentex_dataset/segmentation/enumeration32_train_val_test/train_{size}/masks/{file_name}"


        # Check if both image and mask files exist
        if not (os.path.exists(image_path) and os.path.exists(mask_path)):
            continue

        # read image and mask
        image = Image.open(image_path)
        image = np.array(image)
        mask = Image.open(mask_path)
        mask = np.array(mask)

        # crop image and mask
        for i in range(len(prediction_result["instances"]["classes"])):
            quadrant_id = prediction_result["instances"]["classes"][i]
            quadrant_id = quadrant_remap[quadrant_id]
            bbox = prediction_result["instances"]["boxes"][i]
            bbox = list(map(int, bbox))

            cropped_image_name = f"{file_name[:-4]}_quadrant_{quadrant_id}.png"
            image_names.append(cropped_image_name)

            # crop image and mask
            image_crop = image[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            mask_crop = mask[bbox[1] : bbox[3], bbox[0] : bbox[2]]

            # convert mask to 9 classes
            mask_crop = convert_mask_32_to_9(mask_crop, quadrant_id)

            # flip image and mask to make them look like quadrant 0
            image_crop, mask_crop = flip_quadrants_to_zero(image_crop, mask_crop, quadrant_id)

            # save the flipped image and mask
            Image.fromarray(image_crop).save(os.path.join(output_xrays_dir, cropped_image_name))
            Image.fromarray(mask_crop).save(os.path.join(output_masks_dir, cropped_image_name))

    output_json_path = f"dentex_dataset/segmentation/enumeration9_quadrant_symmetry_augmentation/train_{size}/image_names.json"
    save_json(output_json_path, image_names)

def process_seg_enumeration9_val(quadrant_prediction_path: str):
    quadrant_predictions = load_json(quadrant_prediction_path)
    quadrant_remap = {
        0: 1,
        1: 0,
        2: 2,
        3: 3,
    }
    # remap because category names are different between quadrant and quadrant_enumeration/quadrant_enumeration_disease

    mkdirs("dentex_dataset/segmentation/enumeration9_quadrant_symmetry_augmentation/val/masks")
    mkdirs("dentex_dataset/segmentation/enumeration9_quadrant_symmetry_augmentation/val/xrays")

    image_names = []
    for prediction_result in quadrant_predictions:
        file_name = prediction_result["file_name"]
        image_path = f"dentex_dataset/segmentation/enumeration32_train_val_test/val/xrays/{file_name}"
        mask_path = f"dentex_dataset/segmentation/enumeration32_train_val_test/val/masks/{file_name}"

        # Check if both image and mask files exist
        if not (os.path.exists(image_path) and os.path.exists(mask_path)):
            continue

        # read image and mask
        image = Image.open(image_path)
        image = np.array(image)
        mask = Image.open(mask_path)
        mask = np.array(mask)

        # crop image and mask
        for i in range(len(prediction_result["instances"]["classes"])):
            quadrant_id = prediction_result["instances"]["classes"][i]
            quadrant_id = quadrant_remap[quadrant_id]
            bbox = prediction_result["instances"]["boxes"][i]
            bbox = list(map(int, bbox))

            cropped_image_name = f"{file_name[:-4]}_quadrant_{quadrant_id}.png"
            image_names.append(cropped_image_name)

            # crop image and mask
            image_crop = image[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            mask_crop = mask[bbox[1] : bbox[3], bbox[0] : bbox[2]]

            # convert mask to 9 classes
            mask_crop = convert_mask_32_to_9(mask_crop, quadrant_id)

            # flip image and mask to make them look like quadrant 0
            image_crop, mask_crop = flip_quadrants_to_zero(image_crop, mask_crop, quadrant_id)

            # save the flipped image and mask
            Image.fromarray(image_crop).save(f"dentex_dataset/segmentation/enumeration9_quadrant_symmetry_augmentation/val/xrays/{cropped_image_name}")
            Image.fromarray(mask_crop).save(f"dentex_dataset/segmentation/enumeration9_quadrant_symmetry_augmentation/val/masks/{cropped_image_name}")

    save_json("dentex_dataset/segmentation/enumeration9_quadrant_symmetry_augmentation/val/image_names.json", image_names)

def process_seg_enumeration9_test(quadrant_prediction_path: str):
    quadrant_predictions = load_json(quadrant_prediction_path)
    quadrant_remap = {
        0: 1,
        1: 0,
        2: 2,
        3: 3,
    }
    # remap because category names are different between quadrant and quadrant_enumeration/quadrant_enumeration_disease

    mkdirs("dentex_dataset/segmentation/enumeration9_quadrant_symmetry_augmentation/test/masks")
    mkdirs("dentex_dataset/segmentation/enumeration9_quadrant_symmetry_augmentation/test/xrays")

    image_names = []
    for prediction_result in quadrant_predictions:
        file_name = prediction_result["file_name"]
        image_path = f"dentex_dataset/segmentation/enumeration32_train_val_test/test/xrays/{file_name}"
        mask_path = f"dentex_dataset/segmentation/enumeration32_train_val_test/test/masks/{file_name}"

        # Check if both image and mask files exist
        if not (os.path.exists(image_path) and os.path.exists(mask_path)):
            continue

        # read image and mask
        image = Image.open(image_path)
        image = np.array(image)
        mask = Image.open(mask_path)
        mask = np.array(mask)

        # crop image and mask
        for i in range(len(prediction_result["instances"]["classes"])):
            quadrant_id = prediction_result["instances"]["classes"][i]
            quadrant_id = quadrant_remap[quadrant_id]
            bbox = prediction_result["instances"]["boxes"][i]
            bbox = list(map(int, bbox))

            cropped_image_name = f"{file_name[:-4]}_quadrant_{quadrant_id}.png"
            image_names.append(cropped_image_name)

            # crop image and mask
            image_crop = image[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            mask_crop = mask[bbox[1] : bbox[3], bbox[0] : bbox[2]]

            # convert mask to 9 classes
            mask_crop = convert_mask_32_to_9(mask_crop, quadrant_id)

            # flip image and mask to make them look like quadrant 0
            image_crop, mask_crop = flip_quadrants_to_zero(image_crop, mask_crop, quadrant_id)

            # save the flipped image and mask
            Image.fromarray(image_crop).save(f"dentex_dataset/segmentation/enumeration9_quadrant_symmetry_augmentation/test/xrays/{cropped_image_name}")
            Image.fromarray(mask_crop).save(f"dentex_dataset/segmentation/enumeration9_quadrant_symmetry_augmentation/test/masks/{cropped_image_name}")

    save_json("dentex_dataset/segmentation/enumeration9_quadrant_symmetry_augmentation/test/image_names.json", image_names)


if __name__ == "__main__":
    # process_seg_enumeration9_test("results/enumeration_dataset_quadrant_predictions.json")
    # process_seg_enumeration9_val("results/enumeration_dataset_quadrant_predictions.json")

    initial_subset_size = 80
    increment = 50
    max_training_size = 380

    for size in range(initial_subset_size, max_training_size + 1, increment):
        process_seg_enumeration9_train_range("results/enumeration_dataset_quadrant_predictions.json", size)
        ...