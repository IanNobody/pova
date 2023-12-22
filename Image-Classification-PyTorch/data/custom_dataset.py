from torch.utils.data import Dataset
import torch
import cv2
import numpy as np
from os.path import join
from os import listdir
from os.path import isfile
from os.path import isdir


class CustomDataset(Dataset):
    def __init__(self, img_dir, transform=None, augmentation=None, device=None, custom_model=False, binary=False):
        self.img_root = img_dir

        # self.img = []
        self.custom_model = custom_model

        self.img = [f for f in listdir(img_dir) if f.endswith(".jpg") and not f.endswith(".ref.jpg")]
        self.ann = [t.split("/")[-1].split("-")[0] for t in self.img]

        if binary:
            self.ann_dict = {k: 0 if k == "no_animal" else 1 for k in set(self.ann)}
        else:
            self.ann_dict = {k: v for v, k in enumerate(set(self.ann))}

        self.transform = transform
        self.augmentation = augmentation
        self.device = device

        print(self.num_of_classes())
        print(self.ann_dict)

    def __len__(self):
        return len(self.img)

    def _get_class(self, idx):
        return self.ann_dict[self.ann[idx]]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = join(self.img_root, self.img[idx])
        image = cv2.imread(img_path)

        ann = torch.tensor(self._get_class(idx))

        if self.custom_model:
            bg_path = img_path.replace(".jpg", ".jpg.ref.jpg")
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
        return len(self.ann_dict)

    @staticmethod
    def collate_fn(batch):
        if not len(batch[0][0]) == 2:
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
