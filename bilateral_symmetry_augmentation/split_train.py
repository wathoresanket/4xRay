import os
import sys

# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate one folder back from the current directory
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# Append the parent directory to sys.path
sys.path.append(parent_dir)

import shutil
import random
import json

# # Set paths
# data_dir = "dentex_dataset/segmentation/enumeration32_train_val_test/train"
# output_dir = "dentex_dataset/segmentation/enumeration32_train_val_test"

# Set paths
data_dir = "dentex_dataset/segmentation/enumeration32_train_val_test/train"
output_dir = "dentex_dataset/segmentation/enumeration32_train_val_test"

# Create output directories
os.makedirs(output_dir, exist_ok=True)

# Load image names from JSON file
with open(os.path.join(data_dir, "image_names.json"), "r") as f:
    image_names = json.load(f)

# Shuffle image names
random.shuffle(image_names)

# Define initial subset size and increment
initial_subset_size = 80
increment = 50
max_training_size = 380

# Initialize previous training set
previous_training_set = []

# Loop through increments
for current_training_size in range(initial_subset_size, max_training_size + 1, increment):
    # Create a new folder for the current training size
    current_output_dir = os.path.join(output_dir, f"train_{current_training_size}")
    os.makedirs(current_output_dir, exist_ok=True)

    # Create xrays and masks folders within the current training folder
    xrays_folder = os.path.join(current_output_dir, "xrays")
    masks_folder = os.path.join(current_output_dir, "masks")
    os.makedirs(xrays_folder, exist_ok=True)
    os.makedirs(masks_folder, exist_ok=True)

    # Copy images from the previous training set to the new folder
    for img_name in previous_training_set:
        img_path = os.path.join(data_dir, "xrays", img_name)
        mask_path = os.path.join(data_dir, "masks", img_name)
        shutil.copy(img_path, xrays_folder)
        shutil.copy(mask_path, masks_folder)

    # Add new images to the training set
    new_images = image_names[len(previous_training_set):current_training_size]
    for img_name in new_images:
        img_path = os.path.join(data_dir, "xrays", img_name)
        mask_path = os.path.join(data_dir, "masks", img_name)
        shutil.copy(img_path, xrays_folder)
        shutil.copy(mask_path, masks_folder)

    # Combine previous training set and new images
    current_training_set = previous_training_set + new_images

    # Save shuffled image names to image_names.json in the current output directory
    with open(os.path.join(current_output_dir, "image_names.json"), "w") as f:
        json.dump(current_training_set, f)

    # Update previous training set for the next iteration
    previous_training_set = current_training_set

print("Data splitting completed successfully.")
