import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import os
import numpy as np
from .models.mbsif import MBSIFExtractor

class IrisDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = self._load_samples()
        
    def _load_samples(self):
        samples = []
        classes = os.listdir(self.data_dir)
        for class_id in classes:
            class_dir = os.path.join(self.data_dir, class_id)
            if os.path.isdir(class_dir):
                images = os.listdir(class_dir)
                for img in images:
                    samples.append((os.path.join(class_dir, img), class_id))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_id = self.samples[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            image = self.transform(image)
            
        return image, class_id 