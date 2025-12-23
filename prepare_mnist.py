from torchvision import datasets, transforms
import os
from PIL import Image

# Where to save your dataset
root = "../datasets/mnist"
os.makedirs(root, exist_ok=True)

# Download MNIST
train_data = datasets.MNIST(root=root, train=True, download=True)
val_data = datasets.MNIST(root=root, train=False, download=True)

# Function to save images in YOLO folder structure
def save_mnist_images(data, split="train"):
    split_dir = os.path.join(root, split)
    for i in range(10):
        os.makedirs(os.path.join(split_dir, str(i)), exist_ok=True)
    for idx, (img, label) in enumerate(data):
        img.save(os.path.join(split_dir, str(label), f"{idx}.png"))

# Convert to folders
save_mnist_images(train_data, split="train")
save_mnist_images(val_data, split="val")
print("MNIST dataset saved in YOLO folder structure!")

