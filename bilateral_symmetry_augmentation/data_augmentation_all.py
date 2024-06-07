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

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

class SegmentationPredictor:
    def __init__(self, model, mean=0.458, std=0.173, cuda=True) -> None:
        self.model = model
        self.model.eval()
        self.cuda = cuda
        self.mean = mean
        self.std = std

    def predict(self, image: np.ndarray) -> torch.Tensor:
        origin_shape = image.shape
        image = Image.fromarray(image).resize((256, 256))
        image = F.to_tensor(image)
        image = F.normalize(image, [self.mean], [self.std])
        if self.cuda:
            image = image.cuda()
        image = image.unsqueeze(0)

        predictions = self.model(image)
        predictions = predictions.squeeze(0)
        predictions = torch.argmax(predictions, dim=0, keepdim=True)
        predictions = F.resize(predictions, origin_shape, F.InterpolationMode.NEAREST)
        return predictions.squeeze(0)
    
def data_augmentation_original():
    """
    draw segmentation masks for enumeration32
    """
    dataset_json = load_json("dentex_dataset/origin/quadrant_enumeration/train_quadrant_enumeration.json")
    mkdirs("dentex_dataset/segmentation/enumeration32_bilateral_symmetry_augmentation_all/masks")
    mkdirs("dentex_dataset/segmentation/enumeration32_bilateral_symmetry_augmentation_all/xrays")

    image_names = []
    for image_info in dataset_json["images"]:
        image_names.append(image_info["file_name"])
        # draw mask for each image
        image = Image.open(f"dentex_dataset/origin/quadrant_enumeration/xrays/{image_info['file_name']}")
        mask = Image.new("L", image.size)
        draw = ImageDraw.Draw(mask)

        for annotation in dataset_json["annotations"]:
            if annotation["image_id"] == image_info["id"]:
                points = np.array(annotation["segmentation"]).reshape(-1, 2)
                points = [tuple(point) for point in points]
                # draw polygon, fill with label 1~32
                draw.polygon(points, fill=annotation["category_id_1"] * 8 + annotation["category_id_2"] + 1)

        # save mask and copy image
        mask.save(f"dentex_dataset/segmentation/enumeration32_bilateral_symmetry_augmentation_all/masks/{image_info['file_name']}")
        shutil.copy(
            f"dentex_dataset/origin/quadrant_enumeration/xrays/{image_info['file_name']}",
            f"dentex_dataset/segmentation/enumeration32_bilateral_symmetry_augmentation_all/xrays/{image_info['file_name']}",
        )

    save_json("dentex_dataset/segmentation/enumeration32_bilateral_symmetry_augmentation_all/image_names.json", image_names)

def data_augmentation_left():
    # Check if the file exists
    if not os.path.exists(os.path.join(output_dir, 'image_names.json')):
        # If not, create it
        with open(os.path.join(output_dir, 'image_names.json'), 'w') as f:
            json.dump([], f)  # Initialize it with an empty list

    # Load the list of image names
    with open(os.path.join(output_dir, 'image_names.json'), 'r') as f:
        image_names = json.load(f)

    # For each image in the 'xray' subdirectory
    for image_name in os.listdir(xray_dir):
        # Load the image
        image_path = os.path.join(xray_dir, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Load the model
        model = load_seunet(model_path, 5, cuda=cuda) # out_channels include background

        # Create a predictor
        predictor = SegmentationPredictor(model, cuda=cuda)

        # Predict the image
        prediction = predictor.predict(image)

        # Now `prediction` is a 2D tensor where each value represents the predicted class of the pixel

        # Only show quadrants 0 and 3
        prediction[(prediction != 1) & (prediction != 4)] = 0

        # Convert prediction to numpy for masking
        prediction_np = prediction.cpu().numpy()

        # Create a mask where prediction is greater than 0
        binary_mask = prediction_np > 0

        # Apply the mask to the original image
        image = image * binary_mask

        # Find the rightmost point in the predicted region
        y, x = np.where(binary_mask)
        rightmost_point = max(x)

        # Cut the right side of the image from the rightmost point
        image = image[:, :rightmost_point]

        # Flip the image along y-axis and create a copy
        flipped_image = np.fliplr(image.copy())

        # Join the original and flipped images horizontally
        new_image = np.hstack((image, flipped_image))

        # Extract the base name and extension from the image name
        base_name, extension = os.path.splitext(image_name)

        # Add a suffix to the base name
        new_image_name = f"{base_name}_augmented_left_{extension}"

        # Construct the new image path
        new_image_path = os.path.join(output_xray_dir, new_image_name)

        # Save the new image
        cv2.imwrite(new_image_path, new_image)

        # Add the augmented image name to the list
        image_names.append(new_image_name)

        # Load the mask
        mask_path = os.path.join(masks_dir, image_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply the mask to the original image
        mask = mask * binary_mask

        # Cut the right side of the image from the rightmost point obtained earlier
        mask = mask[:, :rightmost_point]

        # Get the shape of the mask
        height, width = mask.shape

        # Iterate over each pixel in the mask
        for y in range(height):
            for x in range(width):
                # If the grayscale value of the pixel is not between 1-8 or 25-32, make it total black (0)
                if not (1 <= mask[y, x] <= 8 or 25 <= mask[y, x] <= 32):
                    mask[y, x] = 0

        # Flip the mask along y-axis and create a copy
        flipped_mask = np.fliplr(mask.copy())

        # Get the shape of the mask
        height, width = flipped_mask.shape

        # Iterate over each pixel in the mask
        for y in range(height):
            for x in range(width):
                # If the grayscale value of the pixel is between 1-8, add 8 to it
                if 1 <= flipped_mask[y, x] <= 8:
                    flipped_mask[y, x] += 8
                # If the grayscale value of the pixel is between 25-32, subtract 8 from it
                elif 25 <= flipped_mask[y, x] <= 32:
                    flipped_mask[y, x] -= 8

        # Join the original and flipped masks horizontally
        new_mask = np.hstack((mask, flipped_mask))

        # Apply the same name to the mask as the image
        new_mask_name = new_image_name
        
        # Construct the new mask path
        new_mask_path = os.path.join(output_masks_dir, new_mask_name)
        
        # Save the new mask
        cv2.imwrite(new_mask_path, new_mask)

    # Save the updated list of image names
    with open(os.path.join(output_dir, 'image_names.json'), 'w') as f:
        json.dump(image_names, f)


def data_augmentation_right():
    # Check if the file exists
    if not os.path.exists(os.path.join(output_dir, 'image_names.json')):
        # If not, create it
        with open(os.path.join(output_dir, 'image_names.json'), 'w') as f:
            json.dump([], f)  # Initialize it with an empty list

    # Load the list of image names
    with open(os.path.join(output_dir, 'image_names.json'), 'r') as f:
        image_names = json.load(f)

    # For each image in the 'xray' subdirectory
    for image_name in os.listdir(xray_dir):
        # Load the image
        image_path = os.path.join(xray_dir, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Load the model
        model = load_seunet(model_path, 5, cuda=cuda) # out_channels include background

        # Create a predictor
        predictor = SegmentationPredictor(model, cuda=cuda)

        # Predict the image
        prediction = predictor.predict(image)

        # Now `prediction` is a 2D tensor where each value represents the predicted class of the pixel

        # Only show quadrants 1 and 2
        prediction[(prediction != 2) & (prediction != 3)] = 0 

        # Convert prediction to numpy for masking
        prediction_np = prediction.cpu().numpy()

        # Create a mask where prediction is greater than 0
        binary_mask = prediction_np > 0

        # Apply the mask to the original image
        image = image * binary_mask

        # Find the leftmost point in the predicted part
        y, x = np.where(prediction.cpu().numpy() > 0)
        leftmost_point = min(x)

        # Cut the left side of the image from the leftmost point
        image = image[:, leftmost_point:]

        # Flip the image along y-axis and create a copy
        flipped_image = np.fliplr(image.copy())

        # Join the original and flipped images horizontally
        new_image = np.hstack((flipped_image, image))

        # Extract the base name and extension from the image name
        base_name, extension = os.path.splitext(image_name)

        # Add a suffix to the base name
        new_image_name = f"{base_name}_augmented_right_{extension}"

        # Construct the new image path
        new_image_path = os.path.join(output_xray_dir, new_image_name)

        # Save the new image
        cv2.imwrite(new_image_path, new_image)

        # Add the augmented image name to the list
        image_names.append(new_image_name)

        # Load the mask
        mask_path = os.path.join(masks_dir, image_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply the mask to the original image
        mask = mask * binary_mask

        # Cut the left side of the image from the leftmost point
        mask = mask[:, leftmost_point:]

        # Get the shape of the mask
        height, width = mask.shape

        # Iterate over each pixel in the mask
        for y in range(height):
            for x in range(width):
                # If the grayscale value of the pixel is not between 1-8 or 25-32, make it total black (0)
                if not (9 <= mask[y, x] <= 16 or 17 <= mask[y, x] <= 24):
                    mask[y, x] = 0

        # Flip the mask along y-axis and create a copy
        flipped_mask = np.fliplr(mask.copy())

        # Get the shape of the mask
        height, width = flipped_mask.shape

        # Iterate over each pixel in the mask
        for y in range(height):
            for x in range(width):
                # If the grayscale value of the pixel is between 1-8, add 8 to it
                if 9 <= flipped_mask[y, x] <= 16:
                    flipped_mask[y, x] -= 8
                # If the grayscale value of the pixel is between 25-32, subtract 8 from it
                elif 17 <= flipped_mask[y, x] <= 24:
                    flipped_mask[y, x] += 8

        # Join the original and flipped masks horizontally
        new_mask = np.hstack((flipped_mask, mask))

        # Apply the same name to the mask as the image
        new_mask_name = new_image_name
        
        # Construct the new mask path
        new_mask_path = os.path.join(output_masks_dir, new_mask_name)
        
        # Save the new mask
        cv2.imwrite(new_mask_path, new_mask)

    # Save the updated list of image names
    with open(os.path.join(output_dir, 'image_names.json'), 'w') as f:
        json.dump(image_names, f)


def data_augmentation_flip():
    # Check if the file exists
    if not os.path.exists(os.path.join(output_dir, 'image_names.json')):
        # If not, create it
        with open(os.path.join(output_dir, 'image_names.json'), 'w') as f:
            json.dump([], f)  # Initialize it with an empty list

    # Load the list of image names
    with open(os.path.join(output_dir, 'image_names.json'), 'r') as f:
        image_names = json.load(f)

    # For each image in the 'xray' subdirectory
    for image_name in os.listdir(xray_dir):
        # Load the image
        image_path = os.path.join(xray_dir, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Flip the image along y-axis and create a copy
        new_image = np.fliplr(image.copy())

        # Extract the base name and extension from the image name
        base_name, extension = os.path.splitext(image_name)

        # Add a suffix to the base name
        new_image_name = f"{base_name}_augmented_flip_{extension}"

        # Construct the new image path
        new_image_path = os.path.join(output_xray_dir, new_image_name)

        # Save the new image
        cv2.imwrite(new_image_path, new_image)

        # Add the augmented image name to the list
        image_names.append(new_image_name)

        # Load the mask
        mask_path = os.path.join(masks_dir, image_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Get the shape of the mask
        height, width = mask.shape

        # Flip the mask along y-axis and create a copy
        new_mask = np.fliplr(mask.copy())

        # Iterate over each pixel in the mask
        for y in range(height):
            for x in range(width):
                # If the grayscale value of the pixel is between 1-8, add 8 to it
                if 9 <= new_mask[y, x] <= 16:
                    new_mask[y, x] -= 8
                # If the grayscale value of the pixel is between 25-32, subtract 8 from it
                elif 17 <= new_mask[y, x] <= 24:
                    new_mask[y, x] += 8
                # If the grayscale value of the pixel is between 1-8, add 8 to it
                elif 1 <= new_mask[y, x] <= 8:
                    new_mask[y, x] += 8
                # If the grayscale value of the pixel is between 25-32, subtract 8 from it
                elif 25 <= new_mask[y, x] <= 32:
                    new_mask[y, x] -= 8

        # Apply the same name to the mask as the image
        new_mask_name = new_image_name
        
        # Construct the new mask path
        new_mask_path = os.path.join(output_masks_dir, new_mask_name)
        
        # Save the new mask
        cv2.imwrite(new_mask_path, new_mask)

    # Save the updated list of image names
    with open(os.path.join(output_dir, 'image_names.json'), 'w') as f:
        json.dump(image_names, f)


if __name__ == "__main__":

    cuda = True
    model_path = "outputs/output_unet_quadrant_16/epoch_66_loss_0.18880295587910545.pth"
    
    # Input directory
    input_dir = 'dentex_dataset/segmentation/enumeration32/'

    # # for val
    # input_dir = 'dentex_dataset/segmentation/enumeration32_train_val_test/val/'


    # Path to the 'xray' subdirectory
    xray_dir = os.path.join(input_dir, 'xrays')

    # Path to the 'masks' subdirectory
    masks_dir = os.path.join(input_dir, 'masks')


    # Output directory
    output_dir = 'dentex_dataset/segmentation/enumeration32_bilateral_symmetry_augmentation_all/'

    # # for val
    # output_dir = 'dentex_dataset/segmentation/enumeration32_my_data_augmentation/val/'

    # Path to the output 'xray' subdirectory   
    output_xray_dir = os.path.join(output_dir, 'xrays')
    # Create the output directory if it does not exist
    mkdirs(output_xray_dir)

    # Path to the output 'masks' subdirectory
    output_masks_dir = os.path.join(output_dir, 'masks')
    # Create the output directory if it does not exist
    mkdirs(output_masks_dir)

    data_augmentation_original()
    data_augmentation_left()
    data_augmentation_right()
    data_augmentation_flip()
    ...
