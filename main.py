import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import wandb
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import time

# ==========================================
# STEP 0: Initialize wandb
# ==========================================
if wandb.run is not None:
    wandb.finish()

# Try to login to wandb. Ideally, the user should be logged in via terminal (`wandb login`)
# or have WANDB_API_KEY in their environment variables.
try:
    wandb.login()
except Exception as e:
    print(f"Warning: WandB login skipped/failed. Trace: {e}")

wandb.init(
    project="VGG-flowers-v3",
    name="vgg16-transfer-learning-local-run",
    reinit=True,
    config={
        "epochs": 50,
        "batch_size": 16,
        "learning_rate": 0.001,
        "architecture": "VGG16",
        "pretrained": True,
        "input_size": 224
    }
)

config = wandb.config

# =======================
# STEP 1: Data Preparation
# =======================

# HARDCODED PATHS AS REQUESTED
train_dir = "/home/nursena/Downloads/FLowers/flowers/train"
val_dir = "/home/nursena/Downloads/FLowers/flowers/val"

print(f"Train dir: {train_dir}")
print(f"Val dir: {val_dir}")

if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Train directory not found: {train_dir}")
if not os.path.exists(val_dir):
    raise FileNotFoundError(f"Val directory not found: {val_dir}")

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((config.input_size, config.input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((config.input_size, config.input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transforms['train'])
val_dataset = datasets.ImageFolder(root=val_dir, transform=data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

class_names = train_dataset.classes
num_classes = len(class_names)
print(f"Classes found: {class_names}")

# ===========================
# STEP 2: Model Configuration
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == 'cpu':
    print("\n" + "="*50)
    print("WARNING: GPU is not available. Training will be VERY SLOW.")
    print("Please install NVIDIA drivers to enable CUDA acceleration.")
    print("="*50 + "\n")

model = models.vgg16(pretrained=config.pretrained)

for param in model.features.parameters():
    param.requires_grad = False

model.classifier[6] = nn.Linear(4096, num_classes)
model = model.to(device)

# ===========================
# STEP 3: Training Setup
# ===========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, epochs):
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if i % 10 == 0:
               print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        scheduler.step(val_acc)
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{epochs} - Val Acc: {val_acc:.4f} - Time: {epoch_time:.2f}s")
        wandb.log({"epoch": epoch+1, "val_accuracy": val_acc, "loss": running_loss/len(train_loader)})

# Start Training
if __name__ == "__main__":
    try:
        train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, config.epochs)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        wandb.finish()
