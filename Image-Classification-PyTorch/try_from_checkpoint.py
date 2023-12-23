#
# POVa project
# Script for testing the custom model from checkpoint
# Author: Šimon Strýček <xstryc06@vutbr.cz>
#

import argparse
from ModifiedAlexNet import ModifiedAlexNet
from AlexNet import AlexNet
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import cv2
from augmentations import augmentation

def parse_args():
    model_parser = argparse.ArgumentParser(description='Script for testing the custom model from checkpoint')
    model_parser.add_argument('--weights_path', type=str, required=True, help='Path to the weights file')
    model_parser.add_argument('--input_img', type=str, required=True, help='Path to the input image')
    model_parser.add_argument('--background', type=str, required=True, help='Path to the background image')
    return model_parser.parse_args()

def init_model(weights_path, device):
    alexnet = AlexNet(input_channel=3, n_classes=2)
    fc = nn.Sequential(
        nn.Linear(256 * 6 * 6, 4096),  # (batch_size * 4096)
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096),  # (batch_size * 4096)
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 2)
    ).to(device)
    model = ModifiedAlexNet(input_channel=3, alexnet=alexnet, fc=fc).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model

if __name__ == '__main__':
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = init_model(args.weights_path, device)
    model.eval()

    transform = augmentation(image_resolution=224)

    input_img = cv2.imread(args.input_img)
    background = cv2.imread(args.background)

    input_img = transform(input_img).unsqueeze(0).to(device)
    background = transform(background).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_img, background)
        print(output)
        output = torch.argmax(output)
        if output == 0:
            print('The image contains no animal.')
        else:
            print('There is animal in the image.')