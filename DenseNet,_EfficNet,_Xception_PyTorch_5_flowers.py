# Skipped: from google.colab import drive
# Skipped: drive.mount('/content/drive')

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision import models
from torchsummary import summary
import wandb
import os

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================
# STEP 1: Data Preparation
# =======================

# Transforms for training and validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

train_dir = "/home/nursena/Downloads/FLowers/flowers/train"
val_dir = "/home/nursena/Downloads/FLowers/flowers/val"

train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transforms['train'])
val_dataset = datasets.ImageFolder(root=val_dir, transform=data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Training Function
def train_model(model, model_name):
    wandb.init(project=f"{model_name}-flowers", config={
        "epochs": 50,
        "batch_size": 16,
        "learning_rate": 0.001,
        "architecture": model_name,
        "pretrained": True
    })
    config = wandb.config
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
    wandb.watch(model, log="all", log_freq=10)

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        wandb.log({"train_loss": train_loss, "train_accuracy": train_acc})

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total
        wandb.log({"val_accuracy": val_acc})
        print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    wandb.finish()

# Model setup functions
def get_efficientnet():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    for param in model.parameters(): param.requires_grad = False
    for param in model.classifier[1].parameters(): param.requires_grad = True
    return model

def get_densenet():
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, 5)
    for param in model.parameters(): param.requires_grad = False
    for param in model.classifier.parameters(): param.requires_grad = True
    return model

def get_xception():
    from timm import create_model
    model = create_model('xception', pretrained=True, num_classes=5)
    for param in model.parameters(): param.requires_grad = False
    for param in model.get_classifier().parameters(): param.requires_grad = True
    return model

# Main loop
model_dict = {
    # "EfficientNet": get_efficientnet,
    # "DenseNet": get_densenet,
    "Xception": get_xception
}

for name, get_model in model_dict.items():
    model = get_model()
    train_model(model, name)

