import cv2
import numpy as np
import os
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from config.config import Config

def preprocess_image(image):
    """Preprocess a single image"""
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Resize
    image = cv2.resize(image, Config.IMG_SIZE)
    
    # Normalize
    image = image / 255.0
    
    # Enhance contrast
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(np.uint8(l * 255)) / 255.0
    enhanced = cv2.merge([cl, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced

def create_dataset_split():
    """Split dataset into train, validation, and test sets"""
    data_root = Path(Config.DATA_ROOT)
    processed_root = Path(Config.PROCESSED_DATA_ROOT)
    
    # Create processed data directories
    for split in ['train', 'val', 'test']:
        split_dir = processed_root / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True)
    
    # Get disease classes
    disease_classes = [d for d in data_root.iterdir() if d.is_dir()]
    Config.NUM_CLASSES = len(disease_classes)
    
    for disease_dir in disease_classes:
        disease_name = disease_dir.name
        image_files = list(disease_dir.glob('*.[jp][pn][g]'))  # jpg, png, jpeg
        
        if not image_files:
            continue
            
        # Split into train, validation, and test
        train_files, test_files = train_test_split(
            image_files, 
            test_size=(1-Config.TRAIN_SPLIT),
            random_state=42
        )
        
        val_files, test_files = train_test_split(
            test_files,
            test_size=Config.VALIDATION_SPLIT/(1-Config.TRAIN_SPLIT),
            random_state=42
        )
        
        # Process and save images
        for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            split_disease_dir = processed_root / split / disease_name
            split_disease_dir.mkdir(exist_ok=True)
            
            for img_path in files:
                try:
                    # Read and preprocess image
                    img = cv2.imread(str(img_path))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    processed_img = preprocess_image(img)
                    
                    # Save processed image
                    save_path = split_disease_dir / img_path.name
                    cv2.imwrite(str(save_path), cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR) * 255)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    create_dataset_split() 