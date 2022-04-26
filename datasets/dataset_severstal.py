import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import pandas as pd
import cv2


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class MaskGenerator:
    def __init__(self, data_csv):
        self.data_csv = data_csv
        self.df = pd.read_csv(self.data_csv)

    def rle2mask(self, mask_rle, input_shape=(256,1600,4), class_id=1):
        height, width = input_shape[:2]
        
        mask = np.zeros(width * height, dtype=np.uint8)
        if mask_rle is not np.nan:
            s = mask_rle.split()
            array = np.asarray([int(x) for x in s])
            starts = array[0::2]
            lengths = array[1::2]

            for index, start in enumerate(starts):
                begin = int(start - 1)
                end = int(begin + lengths[index])
                mask[begin : end] = class_id
            
        rle_mask = mask.reshape(width, height).T
        return rle_mask 

    def build_mask(self, fname, mask_shape=[256,1600,4]):
        sub_df = self.df[self.df.ImageId == fname][['ImageId', 'ClassId', 'EncodedPixels']]
        
        masks = np.zeros(mask_shape)
        for _,val in sub_df.iterrows():
            masks[:, :, val['ClassId']-1] = self.rle2mask(val['EncodedPixels'], mask_shape, val['ClassId'])
        
        mask = np.max(masks, axis=-1)
        return mask


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Severstal_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.maskGenerator = MaskGenerator(os.path.join(list_dir, 'mask_data.csv'))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        # if self.split == "train":
        slice_name = self.sample_list[idx].strip('\n')
        if 'dataset' in self.data_dir:
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            fileName = slice_name + '.jpg'
            img = cv2.imread(os.path.join(self.data_dir, fileName))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = (img - np.min(img))/(np.max(img) - np.min(img))
            mask = self.maskGenerator.build_mask(fileName)

        sample = {'image': img, 'label': mask}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
