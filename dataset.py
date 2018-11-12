import os
import torch
import numpy
from skimage import io
from skimage import transform
from torch.utils.data import Dataset
from PIL import Image

from settings import CLASSIFIED_TRAIN_DATA_RANDOMTEST, TRAIN_DATA, CLASSIFIED_TRAIN_DATA, CLASSIFIED_TRAIN_DATA_RANDOM, TRAIN_DATA_FACE, TRAIN_DATA_NOT_FACE, TEST_DATA_GOOGLE, CLASSIFIED_TEST_DATA_GOOGLE, TEST_DATA

class FaceDataset(Dataset):
    def __init__(self, train_dir, train_file, transform=None):
        self.train_dir = train_dir
        self.train_file = open(train_file, 'r')
        self.lines = self.train_file.readlines()
        self.transform = transform

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        img_name, is_face = self.lines[index].split(' ')
        image = Image.open(os.path.join(self.train_dir, img_name))
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'is_face': int(is_face[0]), 'image_name': img_name}
        return sample

    def close(self):
        self.train_file.close()


class TestDataset(Dataset):
    def __init__(self, test_dir, test_file, transform=None):
        self.test_dir = test_dir
        self.test_file = open(test_file, 'r')
        self.lines = self.test_file.readlines()
        self.transform = transform

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        img_name, is_face = self.lines[index].split(' ')
        image = Image.open(os.path.join(self.test_dir, img_name))
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'is_face': int(is_face[0]), 'image_name': img_name}
        return sample

    def close(self):
        self.test_file.close()


class ValidDataset(Dataset):
    def __init__(self, transform=None):
        self.train_dir = TRAIN_DATA
        self.train_file = open(CLASSIFIED_TRAIN_DATA_RANDOMTEST, 'r')
        self.lines = self.train_file.readlines()
        self.transform = transform

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        img_name, is_face = self.lines[index].split(' ')
        image = Image.open(os.path.join(TRAIN_DATA, img_name))
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'is_face': int(is_face[0]), 'image_name': img_name}
        return sample

    def close(self):
        self.train_file.close()

class UnknownDataset(Dataset):
    def __init__(self, test_dir, test_file, transform=None):
        self.test_dir = test_dir
        self.test_file = open(test_file, 'r')
        self.lines = self.test_file.readlines()
        self.transform = transform

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        img_name = self.lines[index]
        img_name = img_name[:-1]
        image = Image.open(os.path.join(self.test_dir, img_name))
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'image_name': img_name}
        return sample

    def close(self):
        self.test_file.close()
