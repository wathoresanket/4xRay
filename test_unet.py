import os
import json
import logging
import argparse
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.utils.data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms, InterpolationMode
import torchvision.transforms.functional as ttf
from PIL import Image
import numpy as np
from models.unet.utils import save_state
from models.unet.UNet import UNet
from models.unet.SE_UNet import SEUNet
from models.unet.loss.MultiDiceLoss import MultiDiceLoss


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class SegmentationDataset(Dataset):
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        with open(os.path.join(dataset_dir, "image_names.json"), "r") as f:
            self.image_names = json.load(f)

    def __getitem__(self, index) -> tuple[Image.Image, Image.Image]:
        image_name = self.image_names[index]
        image = Image.open(os.path.join(self.dataset_dir, "xrays", image_name)).convert("L")
        mask = Image.open(os.path.join(self.dataset_dir, "masks", image_name))

        return image, mask

    def __len__(self) -> int:
        return len(self.image_names)


class TransformedDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
    ):
        self.dataset = dataset

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.dataset[index]
        image, mask = self.data_transform(image, mask)
        return image, mask

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def data_transform(
        image: Image.Image, mask: Image.Image = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        convert PIL Image to torch Tensor
        @param image: PIL Image
        @param mask: PIL Image
        """
        # Resize
        image = image.resize((256, 256), Image.BILINEAR)
        mask = mask.resize((256, 256), Image.NEAREST)

        # To tensor
        image = ttf.to_tensor(image)  # shape(1, 256, 256)
        mask = torch.from_numpy(np.array(mask)).long().unsqueeze(0)  # shape(1, 256, 256)

        # Normalize
        image = ttf.normalize(image, [0.458], [0.173])

        mask = mask.squeeze(0)
        return image, mask


def main(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    set_seeds(args.seed)
    cuda = args.cuda

    logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler(os.path.join(output_dir, "testing.log"))
    console_handler = logging.StreamHandler()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    logger.info("Loading dataset...")

    num_classes = args.num_classes
    if args.model == "unet":
        model = UNet(in_channels=1, out_channels=num_classes + 1)
    else:
        model = SEUNet(n_cls=num_classes + 1)
    
    # Load the saved model
    model_path = args.model_path
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from {model_path}")
    else:
        logger.error(f"Model file not found at {model_path}.")
        return

    if cuda:
        model = model.cuda()

    dataset_dir = args.dataset_dir
    dataset = SegmentationDataset(dataset_dir)
    logger.info("Loaded dataset!")
    dataset = TransformedDataset(dataset)

    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    dice_loss_func = MultiDiceLoss()
    ce_loss_func = CrossEntropyLoss()

    # Testing
    logger.info(f"Start testing...")

    model.eval()
    with torch.no_grad():
        test_loss_dice = 0.0
        test_loss_ce = 0.0
        for i, (image, mask) in enumerate(test_loader):
            logger.info(f"Testing batch: {i}/{len(test_loader)}")
            if cuda:
                image = image.cuda()
                mask = mask.cuda()

            pred = model(image)
            loss_dice = dice_loss_func(pred, mask)
            loss_ce = ce_loss_func(pred, mask)

            test_loss_dice += loss_dice.item()
            test_loss_ce += loss_ce.item()

        test_loss_dice /= len(test_loader)
        test_loss_ce /= len(test_loader)
        logger.info(f"Testing loss: {test_loss_dice}, {test_loss_ce}")
        logger.info(f"Testing Dice Loss: {test_loss_dice}")
        logger.info(f"Testing Cross Entropy Loss: {test_loss_ce}")

    logger.info("Done!")


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--model_path", type=str, help="Path to the saved model (.pth file)")
    parser.add_argument("--model", type=str, choices=["unet", "seunet"], default="unet")
    parser.add_argument("--num_classes", type=int, help="number of classes, not including background")
    return parser


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    main(args)
