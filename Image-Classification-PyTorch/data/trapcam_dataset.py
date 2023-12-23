#
#   POVa Project
#	Custom dataset loader for loading images from
#   https://universe.roboflow.com/my-game-pics/my-game-pics
#	Authors:
#	- Šimon Strýček <xstryc06@vutbr.cz>
#   - Kateřina Neprašová <xnepra01@vutbr>
#

from torch.utils.data import Dataset
import torch
import cv2
import numpy as np
from os.path import join
from os import listdir
from os.path import isfile
from os.path import isdir
from os import stat


class TrapcamDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, augmentation=None, device=None):
        self.img_root = img_dir

        self.img = [join(img_dir, f) for f in listdir(img_dir) if isfile(join(img_dir, f)) and f.endswith(".jpg")]
        self.labels = []
        for img in self.img:
            filename = img.split("/")[-1].replace(".jpg", ".txt")
            label_path = join(label_dir, filename)
            with open(label_path, "r") as f:
                if stat(label_path).st_size == 0:
                    self.labels.append(0)
                else:
                    self.labels.append(int(f.readlines()[0].split()[0]) + 1)

        self.transform = transform
        self.augmentation = augmentation
        self.device = device

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img[idx]
        image = cv2.imread(img_path)
        ann = torch.tensor(max(self.labels[idx], 1))

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
        return len(list(set(self.labels)))
