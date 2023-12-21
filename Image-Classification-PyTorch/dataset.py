import torch
import torchvision
from augmentations import augmentation, ContrastiveAugmentation
import torchvision.transforms as transforms
import data.trapcam_dataset as trapcam_dataset
import data.custom_dataset as custom_dataset

class initialize_dataset:
    def __init__(self, image_resolution=224, batch_size=128, MNIST=True):
        self.image_resolution= image_resolution
        self.batch_size=batch_size
        self.MNIST = MNIST

    def load_dataset(self, path, transform=False, custom_model=False):
        #path = "./data"
        #path = './trailcam'
        #path = './custom_dataset'
        #path = './custom_dataset_unprocessed'
        if transform:
            transform = augmentation(image_resolution=self.image_resolution)
        elif self.MNIST:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((self.image_resolution, self.image_resolution)),
                 transforms.RandomHorizontalFlip(), transforms.Normalize(mean=[94.4795, 99.3368, 94.0037], std=[42.7191, 42.3097, 41.7960])])
        else:
            # transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((self.image_resolution, self.image_resolution)),
            #             transforms.RandomHorizontalFlip(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((self.image_resolution, self.image_resolution)),
                 transforms.RandomHorizontalFlip(), transforms.Normalize(mean=[74.9580, 77.5921, 73.6654], std=[50.9340, 51.9717, 50.6036])])

        if self.MNIST:
            train_dataset = torchvision.datasets.CIFAR10(root=path, train=True,
                                                        transform = transform,
                                                        download=True)
            test_dataset = torchvision.datasets.CIFAR10(root=path, train=False,
                                                    transform = transform,
                                                    download=True)
        else:
            # train_dataset = trapcam_dataset.TrapcamDataset(img_dir=path + '/images/train',
            #                                                label_dir=path + '/labels/train', transform=transform)
            # test_dataset = trapcam_dataset.TrapcamDataset(img_dir=path + '/images/test',
            #                                               label_dir=path + '/labels/test', transform=transform)
            train_dataset = custom_dataset.CustomDataset(img_dir=path + '/train',
                                                         transform=transform,
                                                         custom_model=custom_model)
            test_dataset = custom_dataset.CustomDataset(img_dir=path + '/test',
                                                        transform=transform,
                                                        custom_model=custom_model)

        if self.MNIST:
            train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True)
            test_dataloader = torch.utils.data.DataLoader(dataset = test_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                          num_workers=8)
        else:
            train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                           batch_size=self.batch_size,
                                                           shuffle=True,
                                                           collate_fn=custom_dataset.CustomDataset.collate_fn)
            test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                          batch_size=self.batch_size,
                                                          shuffle=True,
                                                          num_workers=8,
                                                          collate_fn=custom_dataset.CustomDataset.collate_fn)

        return train_dataloader, test_dataloader
