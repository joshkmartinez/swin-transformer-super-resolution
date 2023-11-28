import os
import requests
from PIL import Image
import torch
import numpy as np
import pandas as pd
from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim

# Dataset class
class SuperResolutionDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.transform = transform
        self.images = os.listdir(lr_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        lr_image_path = os.path.join(self.lr_dir, self.images[idx])
        hr_image_path = os.path.join(self.hr_dir, self.images[idx])

        lr_image = Image.open(lr_image_path)
        hr_image = Image.open(hr_image_path)

        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        return lr_image, hr_image

# Prepare dataset and dataloader
lr_dir = "train_set/LR"
hr_dir = "train_set/HR"
transform = transforms.Compose([transforms.ToTensor()])

dataset = SuperResolutionDataset(lr_dir, hr_dir, transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

fine_tuned_model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x4-64")
fine_tuned_model.train()


# Optimizer and loss
optimizer = optim.Adam(fine_tuned_model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()

# Training loop
num_epochs = 1  # Define the number of epochs
save_interval = 5  # Save every 2 epochs

all_losses = []

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        lr_images, hr_images = data
        optimizer.zero_grad()

        # Generate SR images
        outputs = fine_tuned_model(lr_images)
        sr_images = outputs.reconstruction

        # Compute loss and backprop
        loss = criterion(sr_images, hr_images)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        running_loss += batch_loss

        print(i, batch_loss)

        # Store the loss for this batch
        all_losses.append({'epoch': epoch+1, 'batch': i+1, 'loss': batch_loss})

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader)}")

    # Save the model
    if (epoch + 1) % save_interval == 0:
        torch.save(fine_tuned_model.state_dict(), f"swin2sr_epoch_{epoch+1}.pth")

df_losses = pd.DataFrame(all_losses)
# print(df_losses)

df_losses.to_csv("training_losses.csv", index=False)

torch.save(fine_tuned_model.state_dict(), f"swin2sr_fine_tuned_FINAL.pth")

print("Training complete")