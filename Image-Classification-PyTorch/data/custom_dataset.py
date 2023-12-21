from torch.utils.data import Dataset
import torch
import cv2
import numpy as np
from os.path import join
from os import listdir
from os.path import isfile
from os.path import isdir


class CustomDataset(Dataset):
    def __init__(self, img_dir, transform=None, augmentation=None, device=None, custom_model=False):
        self.img_root = img_dir

        self.img = []
        self.background = {} if custom_model else None
        for subdir in listdir(img_dir):
            if isdir(join(img_dir, subdir)):
                self.img.extend([join(subdir, f) for f in listdir(join(img_dir, subdir)) if f.endswith(".jpg") and not f.endswith(".ref.jpg")])
                if custom_model:
                    try:
                        self.background[subdir] = next(join(subdir, f) for f in listdir(join(img_dir, subdir)) if f.endswith(".ref.jpg"))
                    except StopIteration:
                        print("No reference image found for {}".format(subdir))

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

        if self.background is not None:
            bg_key = self.img[idx].split("/")[0]
            bg_path = join(self.img_root, self.background[bg_key])
            background = cv2.imread(bg_path)
        else:
            background = None

        if self.augmentation:
            image = self.augmentation(image=image)["image"]
            if background is not None:
                background = self.augmentation(image=background)["image"]

        if self.transform:
            image = self.transform(image)
            if background is not None:
                background = self.transform(background)

        # if self.device:
            # image = image.to(self.device)
            # ann = ann.to(self.device)

        if background is not None:
            sample = ((image, background), ann)
        else:
            sample = (image, ann)

        return sample

    def num_of_classes(self):
        return 2

    @staticmethod
    def collate_fn(batch):
        if not len(batch[0]) == 2:
            images, labels = zip(*batch)
            images = torch.stack(images)
            labels = torch.stack(labels)
            return images, labels
        else:
            img, labels = zip(*batch)
            images = torch.stack([i[0] for i in img])
            backgrounds = torch.stack([i[1] for i in img])
            labels = torch.stack(labels)
            return (images, backgrounds), labels
