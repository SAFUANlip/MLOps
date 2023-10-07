import os
import random

import cv2
from torch.utils.data import Dataset


class MLOpsDataset(Dataset):
    def __init__(self, files_path: str, N_img, transform=None):
        self.N_img = N_img
        self.files_path = files_path
        self.data = random.sample(os.listdir(files_path), k=N_img)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        image_arr = cv2.imread(self.files_path + image, cv2.IMREAD_GRAYSCALE)
        image_arr = cv2.resize(image_arr, (32, 32))
        if self.transform is not None:
            image_arr = self.transform(image_arr)
        return image_arr, image
