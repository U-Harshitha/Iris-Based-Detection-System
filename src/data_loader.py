import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import os
from pathlib import Path
import numpy as np
from config.config import Config

class EyeDiseaseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self._load_samples()
        
    def _load_samples(self):
        for idx, class_dir in enumerate(sorted(self.data_dir.iterdir())):
            if class_dir.is_dir():
                self.class_to_idx[class_dir.name] = idx
                for img_path in class_dir.glob('*.[jp][pn][g]'):
                    self.samples.append((str(img_path), idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
            
        return image, class_idx

def get_data_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = EyeDiseaseDataset(
        os.path.join(Config.PROCESSED_DATA_ROOT, 'train'),
        transform=transform
    )
    
    val_dataset = EyeDiseaseDataset(
        os.path.join(Config.PROCESSED_DATA_ROOT, 'val'),
        transform=transform
    )
    
    test_dataset = EyeDiseaseDataset(
        os.path.join(Config.PROCESSED_DATA_ROOT, 'test'),
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader 