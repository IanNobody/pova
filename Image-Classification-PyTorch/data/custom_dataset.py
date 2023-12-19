from torch.utils.data import Dataset
import torch
import cv2
import numpy as np
from os.path import join
from os import listdir
from os.path import isfile
from os.path import isdir


class CustomDataset(Dataset):
    def __init__(self, img_dir, transform=None, augmentation=None, device=None):
        self.img_root = img_dir

        self.img = [f for f in listdir(img_dir) if isfile(join(img_dir, f)) and f.endswith(".jpg")]
        self.ann = [t.split("/")[-1].split("-")[0] for t in self.img]

        self.transform = transform
        self.augmentation = augmentation
        self.device = device

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = join(self.img_root, self.img[idx])
        image = cv2.imread(img_path)
        ann = torch.tensor(0 if self.ann[idx] == "no_animal" else 1)

        if self.augmentation:
            image = self.augmentation(image=image)["image"]

        if self.transform:
            image = self.transform(image)

        # if self.device:
            # image = image.to(self.device)
            # ann = ann.to(self.device)

        sample = (image, ann)
        return sample

    def num_of_classes(self):
        return 2
