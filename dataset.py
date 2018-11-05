import os
import torch
import numpy
from skimage import io
from skimage import transform
from torch.utils.data import Dataset

from settings import CLASSIFIED_TRAIN_DATA_RANDOMTEST, TRAIN_DATA, CLASSIFIED_TRAIN_DATA, CLASSIFIED_TRAIN_DATA_RANDOM, TRAIN_DATA_FACE, TRAIN_DATA_NOT_FACE, TEST_DATA_GOOGLE, CLASSIFIED_TEST_DATA_GOOGLE, TEST_DATA

class FaceDataset(Dataset):
    def __init__(self, transform=None):
        self.train_dir = TRAIN_DATA
        self.train_file = open(CLASSIFIED_TRAIN_DATA_RANDOM, 'r')
        self.lines = self.train_file.readlines()
        self.transform = transform

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        img_name, is_face = self.lines[index].split(' ')
        image = io.imread(os.path.join(TRAIN_DATA, img_name))
        sample = {'image': image, 'is_face': is_face, 'image_name': img_name}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def close(self):
        self.train_file.close()


class TestDataset(Dataset):
    def __init__(self, test_file, transform=None):
        self.train_dir = TEST_DATA
        self.train_file = open(test_file, 'r')
        self.lines = self.train_file.readlines()
        self.transform = transform

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        img_name, is_face = self.lines[index].split(' ')
        image = io.imread(os.path.join(TEST_DATA, img_name))
        sample = {'image': image, 'is_face': is_face, 'image_name': img_name}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def close(self):
        self.train_file.close()


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, is_face, img_name = sample['image'], sample['is_face'], sample['image_name']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # add dimension ?
        # https://discuss.pytorch.org/t/expected-4d-tensor-as-input-got-3d-tensor-instead/6447/2
        # rip transpose
        # image.transpose((2,0,1))
        image = image[None, :]
        return {'image': torch.from_numpy(image),
                'is_face': int(is_face[0]),
                'image_name': img_name}


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
        image = io.imread(os.path.join(TRAIN_DATA, img_name))
        sample = {'image': image, 'is_face': is_face, 'image_name': img_name}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def close(self):
        self.train_file.close()
