import numpy as np
import pandas as pd
import os
import re
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from pytorch_lightning import LightningDataModule

import torch
import random
import numpy as np
from PIL import Image
from PIL import ImageOps
from torchvision.transforms import ColorJitter


class Normalize(object):
    """Normalizes image with range of 0-255 to 0-1.
    """

    def __init__(self, min_val=0, max_val=255):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, sample: dict):
        image = sample['image']
        image -= self.min_val
        image /= (self.max_val - self.min_val)
        image = torch.clamp(image, 0, 1)

        sample.update({
            'image': image,
        })

        return sample


class ZScoreNormalize(object):

    def __call__(self, sample):
        image = sample['image']
        mean = image.mean()
        std = image.std()
        image = image.float()
        image = (image - mean) / std

        sample.update({
            'image': image,
        })

        return sample


class ToImage(object):

    def __call__(self, sample):
        # assert 'label' not in sample.keys()
        image = sample['image']

        sample.update({
            'image': Image.fromarray(image),
        })

        return sample


class ToTensor(object):

    def __call__(self, sample: dict):
        image = sample['image']

        if type(image) == Image.Image:
            image = np.asarray(image)

        if image.ndim == 2:
            image = image[np.newaxis, ...]

        #image = torch.from_numpy(image).float()
        sample.update({
            'image': image,
        })

        if 'label' in sample.keys():
            label = sample['label']

            if label.ndim == 2:
                label = label[np.newaxis, ...]

            #label = torch.from_numpy(label).int()
            sample.update({
                'label': label,
            })

        return sample


class RandomHorizontalFlip(object):

    def __call__(self, sample: dict):
        assert 'label' not in sample.keys()
        image = sample['image']

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        sample.update({
            'image': image,
        })

        return sample


class RandomVerticalFlip(object):

    def __call__(self, sample: dict):
        assert 'label' not in sample.keys()
        image = sample['image']

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)

        sample.update({
            'image': image,
        })

        return sample


class RandomRotate(object):

    def __init__(self, degree=20):
        self.degree = degree

    def __call__(self, sample: dict):
        assert 'label' not in sample.keys()
        image = sample['image']

        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        image = image.rotate(rotate_degree, Image.BILINEAR)

        sample.update({
            'image': image,
        })

        return sample


class RandomScale(object):

    def __init__(self, mean=1.0, var=0.05, image_fill=0):
        self.mean = mean
        self.var = var
        self.image_fill = image_fill

    def __call__(self, sample: dict):
        assert 'label' not in sample.keys()
        image = sample['image']
        base_size = image.size

        scale_factor = random.normalvariate(self.mean, self.var)

        size = (
            int(base_size[0] * scale_factor),
            int(base_size[1] * scale_factor),
        )

        image = image.resize(size, Image.BILINEAR)

        if scale_factor < 1.0:
            pad_h = base_size[0] - image.size[0]
            pad_w = base_size[1] - image.size[1]
            ori_h = random.randint(0, pad_h)
            ori_w = random.randint(0, pad_w)

            image = ImageOps.expand(
                image,
                border=(ori_h, ori_w, pad_h - ori_h, pad_w - ori_w),
                fill=self.image_fill
            )

        else:
            ori_h = random.randint(0, image.size[0] - base_size[0])
            ori_w = random.randint(0, image.size[1] - base_size[1])
            image = image.crop((
                ori_h, ori_w,
                ori_h + base_size[0], ori_w + base_size[1]
            ))

        sample.update({
            'image': image,
        })

        return sample


class RandomColorJitter(object):
    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.3):
        self.filter = ColorJitter(brightness, contrast, saturation)

    def __call__(self, sample: dict):
        image = sample['image']

        image = image.convert('RGB')
        image = self.filter(image)
        image = image.convert('L')

        sample.update({
            'image': image,
        })

        return sample


class RandomSliceSelect(object):
    def __init__(self, threshold=1, max_iter=10):
        self.threshold = threshold
        self.max_iter = max_iter

    def __call__(self, sample: dict):
        image = sample['image']

        z_max = image.shape[2]
        mean = 0.0
        n_iter = 0

        while n_iter < self.max_iter:
            selected_z = random.randint(0, z_max - 1)
            selected = image[..., selected_z]
            mean = np.mean(selected)

            if mean > self.threshold:
                break

            n_iter += 1

        sample.update({
            'image': selected,
        })

        return sample


class CKBrainMetDataset(Dataset):

    def __init__(self, mode, patient_paths, transform, image_size):
        super().__init__()
        assert mode in ['train', 'test']
        """
        if mode == train       -> output only normal images without label
        if mode == test        -> output both normal and abnormal images with label
        """
        self.mode = mode
        self.patient_paths = patient_paths
        self.transform = transform
        self.image_size = image_size
        self.files = self.build_file_paths(self.patient_paths)

    def build_file_paths(self, patient_paths):

        files = []

        for patient_path in patient_paths:
            file_paths = glob(os.path.join(patient_path + "/*" + 'flair' + ".npy")) #指定のスライスのパスを取得
            for file_path in file_paths:
                
                if 'Abnormal' in file_path:
                    class_name = 'Abnormal'
                else:
                    #assert 'normal' in file_name
                    class_name = 'Normal'

                patient_id = patient_path.split('/')[-1]
                file_name = file_path.split('/')[-1]
                study_name = self.get_study_name(patient_path)
                slice_num = self.get_slice_num(file_name)
                path_name = patient_path
                

                if self.mode == 'train':
                    files.append({
                        'image': file_path,
                        'path_name': path_name,
                        'file_name': file_name,
                        'patient_id': patient_id,
                        'class_name': class_name,
                        'study_name': study_name,
                        'slice_num': slice_num,
                    })

                elif self.mode == 'test' or self.mode == 'test_normal':
                    label_path = self.get_label_path(file_path)

                    files.append({
                        'image': file_path,
                        'path_name': path_name,
                        'file_name': file_name,
                        'label': label_path,
                        'patient_id': patient_id,
                        'class_name': class_name,
                        'study_name': study_name,
                        'slice_num': slice_num,
                    })

        return files

    def get_study_name(self, patient_path):
        study_name = patient_path.split('/')[-3]
        return study_name
    
    def get_slice_num(self, file_name):
        n = re.findall(r'\d+', file_name) #image_fileのスライス番号の取り出し
        return n[-1]

    def get_label_path(self, file_path):
        file_path = file_path.replace(self.config.dataset.select_slice, 'seg')
        return file_path

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = np.load(self.files[index]['image'])
        image = np.flipud(np.transpose(image))

        sample = {
            'image': image.astype(np.float32),
            'path_name': self.files[index]['path_name'],
            'file_name': self.files[index]['file_name'],
            'patient_id': self.files[index]['patient_id'],
            'class_name': self.files[index]['class_name'],
            'study_name': self.files[index]['study_name'],
            'slice_num': self.files[index]['slice_num'],
        }

        return sample

def get_patient_paths(base_dir_path):
        patient_ids = os.listdir(base_dir_path)
        return [os.path.join(base_dir_path, p) for p in patient_ids]


val_transform = transforms.Compose([
                    ToImage(),
                    ToTensor(),
                ])

root_dir_path = './data/brats_separated'

train_patient_paths = get_patient_paths(os.path.join(root_dir_path, 'MICCAI_BraTS_2019_Data_Val_Testing/Normal'))
train_dataset = CKBrainMetDataset(mode='train', patient_paths=train_patient_paths, transform=val_transform, image_size=256)

for i in tqdm(range(len(train_dataset))):
    if np.count_nonzero(train_dataset[i]['image'] <= 0) <=57000:
        os.makedirs(train_dataset[i]['path_name'].replace("brats_separated", "fill_data_57000"), exist_ok=True)
        np.save(train_dataset[i]['path_name'].replace("brats_separated", "fill_data_57000") + '/' + train_dataset[i]['file_name'], train_dataset[i]['image'])