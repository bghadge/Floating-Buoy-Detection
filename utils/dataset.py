# from __future__ import print_function, division
import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class CustomDataset(Dataset):
    """
        Create a custom dataset from a txt file that contains names of images.
    """

    def __init__(self, txt_file_path, dataset_dir, transform=None):
        """
        Args:
            txt_file (string): Path to the train/val/test txt file.
            dataset_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.landmarks_frame = pd.read_csv(csv_file)
        self.image_names = pd.read_table(txt_file_path, sep="\n", header=None)
        self.dataset_dir = dataset_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.dataset_dir + self.image_names.iloc[idx,0])
        info_path = os.path.splitext(img_path)[0] + ".txt"
        
        bgr_image = cv2.imread(img_path)
        image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        img_info = pd.read_table(info_path, sep=" ", header=None)
        img_info = np.array(img_info)
        img_info = img_info.astype(np.float).reshape(-1, 5)

        sample = {'image': image, 'bboxes': img_info}

        if self.transform:
            toPIL = transforms.ToPILImage()
            toTensor = transforms.ToTensor()
            sample['image'] = toPIL(sample['image'])
            sample['image'] = self.transform(sample['image'])
            sample['bboxes'] = toTensor(sample['bboxes'])

        return sample