#!/usr/bin/env python
# coding: utf-8
# License: Apache 2.0
# Author: Calvin Winkowski

from __future__ import print_function, division

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets import VisionDataset
from PIL import Image
import time
import os
import os.path
import copy

# Parameters of interest
fc_int = 300

class_strings = [
    "Center stack, Cup holder - console",
    "Forward, Instrument cluster, Left windshield",
    "Left window / mirror",
    "Rearview mirror",
    "Right window / mirror",
    "Right windshield"
]

import torch
from torchvision import datasets

def do_run(model_path, image):
    slug = os.path.basename(model_path).rsplit('.', 1)[0]

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = Image.open(image)
    image_t = data_transform(image)
    image_t = image_t.unsqueeze(0)

    image_t = image_t.to(device)

    class_names = [1, 3, 4, 6, 7, 8]


    # Finetuning the convnet
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    classifier = nn.Sequential(nn.Linear(num_ftrs, fc_int),
        nn.ReLU(),
        nn.Dropout(p = 0.2),
        nn.Linear(fc_int ,len(class_names)),
        nn.LogSoftmax(dim = 1))
    model_ft.fc = classifier

    checkpoint = None
    if torch.cuda.is_available():
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    model_ft.load_state_dict(checkpoint)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()


    model_ft.eval()   # Set model to evaluate mode
    outputs = model_ft(image_t)
    _, preds = torch.max(outputs, 1)
    preds = preds.cpu().data.numpy()

    print(outputs)
    print(preds)
    print(class_strings[preds[0]])

if __name__ == '__main__':
    #do_run()
    import argparse

    print(f"Inference on {sys.argv[1]}")
    do_run(sys.argv[1], sys.argv[2])
