import os
import shutil
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

def prepare_tiny_imagenet_data():
    # Download and unzip the dataset
    !wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
    !unzip tiny-imagenet-200.zip -d tiny-imagenet

    # Organize the validation set
    with open('tiny-imagenet/tiny-imagenet-200/val/val_annotations.txt') as f:
        for line in f:
            fn, cls, *_ = line.split('\t')
            os.makedirs(f'tiny-imagenet/tiny-imagenet-200/val/{cls}', exist_ok=True)
            shutil.copyfile(f'tiny-imagenet/tiny-imagenet-200/val/images/{fn}', f'tiny-imagenet/tiny-imagenet-200/val/{cls}/{fn}')

    shutil.rmtree('tiny-imagenet/tiny-imagenet-200/val/images')

def get_tiny_imagenet_datasets():
    transform = T.Compose([
        T.Resize((224, 224)),  # Resize to fit the input dimensions of the network
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/train', transform=transform)
    val_dataset = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/val', transform=transform)

    return train_dataset, val_dataset