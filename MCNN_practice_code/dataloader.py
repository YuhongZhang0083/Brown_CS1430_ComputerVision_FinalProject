
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize

class CrowdDataset(Dataset):
    def __init__(self, img_dir, gt_dir, transform=None):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.img_filenames = [filename for filename in os.listdir(img_dir) if filename.endswith('.jpg')]

        self.resize_transform = Compose([
            Resize((480, 640)),
            ToTensor()
        ])

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        try:
            img_filename = os.path.join(self.img_dir, self.img_filenames[idx])
            img = Image.open(img_filename)

            gt_filename = os.path.join(self.gt_dir, self.img_filenames[idx].replace('.jpg', '_density.png'))
            # gt = np.array(Image.open(gt_filename))
            gt = Image.open(gt_filename)

            # Convert the ground truth to grayscale
            gt = gt.convert("L")
            
            # Ensure that the ground truth has the same number of channels as the input image
            if gt.mode != img.mode:
                gt = gt.convert(img.mode)

            img = self.resize_transform(img)
            gt = self.resize_transform(gt)

            # Add print statements to display the shapes of images and ground truth
            # print(f"Image {idx} shape: {img.shape}")
            # print(f"GT {idx} shape: {gt.shape}")

            return img, gt
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            raise e

