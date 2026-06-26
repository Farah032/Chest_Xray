import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torchvision.models import DenseNet



model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(50176,14)
            )

images = torch.randn(32, 1, 224, 224)
labels = torch.randint(0, 2, (32, 14)).float()
dataset = TensorDataset(images, labels)

train_loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(2):
    for image, label in train_loader:
        optimizer.zero_grad()
        output =  model(image)
        loss = criterion(output,label)
        loss.backward()
        optimizer.step()
    print(loss.item())