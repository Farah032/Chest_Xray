import pandas as pd
import torch
import os
from PIL import Image 
from torch.utils.data import Dataset
from torchvision import transforms

import os

# Detect if running on Kaggle or local Mac
IS_KAGGLE = os.path.exists('/kaggle')

if IS_KAGGLE:
    # Kaggle paths - we'll use Kaggle's chest X-ray dataset
    IMG_DIR = '/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL'
    CSV_PATH = None  # Kaggle dataset doesn't have this exact CSV
else:
    # Local Mac paths
    IMG_DIR = '/Users/farahjabeen/Desktop/XRAY_PROJECT/data/NIH/images_001/images'
    CSV_PATH = '/Users/farahjabeen/Desktop/XRAY_PROJECT/data/NIH/Data_Entry_2017.csv'

def encode_labels(label_string, classes):
    encoded = torch.zeros(len(classes), dtype=torch.float32)
    labels = label_string.split("|")
    for label in labels:
        if label in classes:
            idx = classes.index(label)
            encoded[idx] = 1.0
    return encoded   

class ChestXrayDataset(Dataset):
    def __init__(self, csv_path=None, img_dir=None, classes=None, transform=None):
        if csv_path is None:
            csv_path = CSV_PATH
        if img_dir is None:
            img_dir = IMG_DIR

        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir 
        self.transform = transform
        self.classes = classes

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        filename = self.df.iloc[idx]['Image Index']
        label = self.df.iloc[idx]['Finding Labels']
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert('L')
        if self.transform is not None:
            image = self.transform(image)
        image = image.repeat(3, 1, 1) 
        labels = encode_labels(label, self.classes)
        return image, labels

