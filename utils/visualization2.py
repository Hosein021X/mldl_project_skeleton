import matplotlib.pyplot as plt
import numpy as np

def denormalize(image):
    image = image.to('cpu').numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image, 0, 1)
    return image

def visualize_classes(dataloader, num_classes=10):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # 2 rows, 5 columns for 10 classes
    classes_sampled = []
    found_classes = 0

    for idx, (inputs, classes) in enumerate(dataloader):
        for img, label in zip(inputs, classes):
            if label.item() not in classes_sampled:
                ax = axes[found_classes // 5, found_classes % 5]  # Access subplot
                ax.axis("off")

                # Convert tensor to numpy image
                img = img.permute(1, 2, 0)  # Convert (C, H, W) → (H, W, C) for plt.imshow()
                ax.imshow(img.numpy())

                ax.set_title(f"Class: {label.item()}")
                classes_sampled.append(label.item())
                found_classes += 1

                if found_classes == num_classes:  # Stop when 10 classes are sampled
                    break
        if found_classes == num_classes:
            break

    plt.tight_layout()
    plt.show()