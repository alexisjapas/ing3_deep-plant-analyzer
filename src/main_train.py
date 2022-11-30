import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from FeaturesPredictionDataset import FeaturesPredictionDataset
from models import FlatCNN
from train import train


# hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")
print(torch.cuda.get_device_name())
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())

# dataset
train_data_path = "../dataset/Train.csv"
crop_size = 84
train_dataset = FeaturesPredictionDataset(train_data_path, crop_size)

# dataloader
batch_size = 4
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True, pin_memory=True)
for inputs, targets in train_dataloader: # DEBUG
    print(inputs.shape, targets.shape)
    break

# model
model = FlatCNN(crop_size, 1, 64, 20, 4).to(device)
print(model)

# train
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 100
for e in range(epochs):
    print(f"Epoch {e+1}")
    epoch_loss = train(train_dataloader, model, loss_fn, optimizer, device)
    print(f"Epoch loss: {epoch_loss}")
    print("-----------------------------")
print("Training finished")

# save model
torch.save(model, 'model.pth')

