import pandas as pd
import torch
import os
from PIL import Image 
from torch.utils.data import Dataset
from torchvision import transforms

def encode_labels(label_string, classes):
    encoded = torch.zeros(len(classes), dtype=torch.float32)
    labels = label_string.split("|")
    for label in labels:
        if label in classes:
            idx = classes.index(label)
            encoded[idx] = 1.0
    return encoded   

class ChestXrayDataset(Dataset):
    def __init__(self, csv_path, img_dir, classes, transform=None):
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

# That's it. NOTHING else. No dataset creation, no printing.