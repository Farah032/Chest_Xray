import pandas as pd
import torch
import os
from PIL import Image 
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

def encode_labels(label_string, classes):
    encoded = torch.zeros(len(classes), dtype=torch.float32)
    labels = label_string.split("|")
    for label in  labels:
        if label in classes:
            idx = classes.index(label)
            encoded[idx] = 1.0
    return encoded   

class ChestXrayDataset(Dataset):
    def __init__(self,csv_path,img_dir,classes,transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir 
        self.transform = transform
        self.classes= classes

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        filename = self.df.iloc[idx]['Image Index']
        label = self.df.iloc[idx]['Finding Labels']
        img_path = os.path.join(self.img_dir , filename)
        image = Image.open(img_path).convert('L')

        if self.transform is not None:
            image = self.transform(image)
   
        labels = encode_labels(label,self.classes)

        
        return image, labels
        




if __name__ == "__main__":
    
    classes = ["Atelectasis", 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
    
    print(encode_labels("Atelectasis|Effusion", classes))
    print(encode_labels("No Finding", classes))
    print(encode_labels("Cardiomegaly", classes))

    img_dir = '/Users/farahjabeen/Desktop/XRAY_PROJECT/data/NIH/images_001/images'
    csv_path = '/Users/farahjabeen/Desktop/XRAY_PROJECT/data/NIH/Data_Entry_2017.csv'

    
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])

    ])


    available = set(os.listdir(img_dir))
    df_full = pd.read_csv(csv_path)
    df_filtered = df_full[df_full['Image Index'].isin(available)]
    filtered_csv_path = '/Users/farahjabeen/Desktop/XRAY_PROJECT/data/NIH/filtered_entry.csv'
    df_filtered.to_csv(filtered_csv_path, index=False)
    print("Filtered images:", len(df_filtered))
    
    dataset = ChestXrayDataset(filtered_csv_path, img_dir, classes, transform)

print("Total images:", len(dataset))
print("First label raw:", dataset.df.iloc[0]['Finding Labels'])
print("First label encoded:", encode_labels(dataset.df.iloc[0]['Finding Labels'], classes))

image, label = dataset[0]
print("Image shape:", image.shape)
print("Label:", label)

loader = DataLoader(dataset, batch_size=32, shuffle=True , num_workers=0)
image, label = next(iter(loader))
print(image.shape, label.shape)


train_size = 3499
val_size = 749
test_size = 751
train_set, val_set, test_set = random_split(dataset,[train_size, val_size, test_size])

print(len(train_set),len(val_set),len(test_set))