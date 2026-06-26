import pandas as pd
import torch
import os
from PIL import Image 
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from preprocessing.chestXray_dataset import ChestXrayDataset 
import torch.nn as nn
from torchvision.models import densenet121

if __name__ == "__main__":
    
    classes = ["Atelectasis", 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
    

    img_dir = '/Users/farahjabeen/Desktop/XRAY_PROJECT/data/NIH/images_001/images'
    csv_path = '/Users/farahjabeen/Desktop/XRAY_PROJECT/data/NIH/Data_Entry_2017.csv'

    
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])

    ])

    # ===== CREATE FILTERED CSV =====
    available = set(os.listdir(img_dir))
    df_full = pd.read_csv(csv_path)
    df_filtered = df_full[df_full['Image Index'].isin(available)]
    filtered_csv_path = '/Users/farahjabeen/Desktop/XRAY_PROJECT/data/NIH/filtered_entry.csv'
    df_filtered.to_csv(filtered_csv_path, index=False)
    print(f"Filtered CSV created: {len(df_filtered)} images")


# ===== CREATE DATASET AND LOADER =====
    dataset = ChestXrayDataset(filtered_csv_path, img_dir, classes, transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

# ===== MODEL, LOSS, OPTIMIZER =====
    '''
    model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(50176, 14))
    '''
    model = densenet121(pretrained=True)
    model.classifier = nn.Linear(in_features=1024,out_features=14)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ===== TWO-EPOCH TRAINING LOOP =====
    for epoch in range(2):
        for images, labels in loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
