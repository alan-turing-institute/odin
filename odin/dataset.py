import mrcfile
import starfile

import matplotlib.pyplot as plt
from matplotlib import patches

import os
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN

from torch.utils.data import DataLoader


class EmpiarImageDataset(Dataset):
    def __init__(self, folder_names, transform=None, target_transform=None):
        df = pd.DataFrame(columns=['mrcfolder', 'mrcfile', 'starfolder', 'starfile'])
        for folder_name in folder_names:
            mrcpath, starpath = make_path(folder_name)
            onlymrc = [f for f in listdir(mrcpath) if isfile(join(mrcpath, f))]
            for i in range(0, len(onlymrc)):
                df2 = pd.DataFrame([[mrcpath, onlymrc[i], starpath, onlymrc[i].split(".")[0] + "_autopick.star"]],
                                   columns=['mrcfolder', 'mrcfile', 'starfolder', 'starfile'])
                df = pd.concat([df, df2], axis=0, ignore_index=True)

        self.full_dirs = df
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.full_dirs)

    def __getitem__(self, idx):
        # Get image
        img_path = os.path.join(self.full_dirs.iloc[idx, 0], self.full_dirs.iloc[idx, 1])
        image = torch.tensor(mrcfile.read(img_path))

        # Get target, aka, bouding boxes from starfile filament description
        target_path = os.path.join(self.full_dirs.iloc[idx, 2], self.full_dirs.iloc[idx, 3])
        star_coords = starfile.read(target_path)
        boxes = get_boxes(star_coords)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
        # target["image_id"] = [self.full_dirs.iloc[idx, 0]]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target


def make_path(folder_name):
    mrcpath = "10943/mrc/" + folder_name + "/"
    starpath = "10943/star/" + folder_name + "/Data/"
    return mrcpath, starpath


def get_boxes(star_coords):
    i = 1
    filaments = []
    while i < star_coords.shape[0]:
        start_x = star_coords.iloc[i - 1].rlnCoordinateX
        start_y = star_coords.iloc[i - 1].rlnCoordinateY
        end_x = star_coords.iloc[i].rlnCoordinateX
        end_y = star_coords.iloc[i].rlnCoordinateY
        filaments.append([start_x, start_y, end_x, end_y])
        i = i + 2

    boxes = []
    for filament in filaments:
        # n_points = int(np.sqrt((filament[2] - filament[0]) ** 2 + (filament[3] - filament[1]) ** 2) / 0.02)
        n_points = int(np.sqrt(abs(filament[2] - filament[0]) + abs(filament[3] - filament[1])) / 2)
        # n_points = 20
        dx = np.linspace(filament[0], filament[2], n_points)
        dy = np.linspace(filament[1], filament[3], n_points)

        box_buffer = 32
        for i in range(0, len(dx)):
            temp_box = [dx[i] - box_buffer, dy[i] - box_buffer, dx[i] + box_buffer, dy[i] + box_buffer]
            boxes.append(temp_box)

    return boxes
