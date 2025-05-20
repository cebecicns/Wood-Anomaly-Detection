import os
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


CLASS_NAMES = ['wood']

class MVTecDataset(Dataset):
    def __init__(self, dataset_path='./data/wood_dataset', class_name='wood', is_train=True,
                 resize=256, cropsize=256):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = os.path.normpath(dataset_path)
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize


        self.x, self.y, self.mask = self.load_dataset_folder()

        self.transform_x = T.Compose([T.Resize(resize, Image.LANCZOS),
                                      T.CenterCrop(cropsize),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])
        self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),
                                         T.CenterCrop(cropsize),
                                         T.ToTensor()])


    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        if y == 0:
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = ['good'] if self.is_train else ['good', 'defect']
        
        for img_type in img_types:
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.exists(img_type_dir):
                continue

            img_files = sorted([f for f in os.listdir(img_type_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
            
            # EĞİTİM verisi için varsayılan etiketler (0=normal)
            if self.is_train:
                y.extend([0] * len(img_files))  # Tüm eğitim görselleri normal (0)
                mask.extend([None] * len(img_files))  # Eğitim için maske yok
            else:
                # TEST verisi işlemleri
                if img_type == 'good':
                    y.extend([0] * len(img_files))
                    mask.extend([None] * len(img_files))
                else:  # defect
                    y.extend([1] * len(img_files))
                    mask_paths = []
                    for f in img_files:
                        mask_path = os.path.join(gt_dir, 'defect', f.replace('.jpg', '_mask.jpg').replace('.png', '_mask.png'))
                        mask_paths.append(mask_path if os.path.exists(mask_path) else None)
                    mask.extend(mask_paths)

            x.extend([os.path.join(img_type_dir, f) for f in img_files])

        return x, y, mask