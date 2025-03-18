import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import requests
from zipfile import ZipFile
from io import BytesIO

def download_and_extract_tiny_imagenet():
    # Define the path to the dataset
    dataset_path = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'

    # Send a GET request to the URL
    response = requests.get(dataset_path)
    if response.status_code == 200:
        # Open the downloaded bytes and extract them
        with ZipFile(BytesIO(response.content)) as zip_file:
            zip_file.extractall('/dataset')
        print('Download and extraction complete!')

def get_tiny_imagenet_datasets():
    # Define transformations for the dataset
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the dataset
    train_dataset = ImageFolder(root='/dataset/tiny-imagenet-200/train', transform=transform)
    test_dataset = ImageFolder(root='/dataset/tiny-imagenet-200/test', transform=transform)

    return train_dataset, test_dataset

def get_tiny_imagenet_dataloaders(batch_size=32):
    train_dataset, test_dataset = get_tiny_imagenet_datasets()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader