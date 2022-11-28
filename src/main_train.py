import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from FeaturesPredictionDataset import FeaturesPredictionDataset
from models import FlatCNN
from train import train


#### TRAINING
# Dataset
train_data_path = "../dataset/Train.csv"
crop_size = 84
train_dataset = FeaturesPredictionDataset(train_data_path, crop_size)


#### MODEL
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")
model = FlatCNN(crop_size, 1, 32, 8, 4).to(device)
print(model)


# Dataloader
batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True, pin_memory=True)
for inputs, targets in train_dataloader: # DEBUG
    print(inputs.shape, targets.shape)
    break

# Train
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 100
for e in range(epochs):
    print(f"Epoch {e+1}")
    epoch_loss = train(train_dataloader, model, loss_fn, optimizer, device)
    print(f"Epoch loss: {epoch_loss}")
    print("-----------------------------")
print("Training finished")


#### SAVING MODEL
torch.save(model, 'model.pth')


#### TESTING

