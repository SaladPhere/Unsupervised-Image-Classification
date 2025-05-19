from torchvision.datasets import STL10
import os

def download_stl10():
    print("Downloading STL-10 dataset...")
    print("This might take a few minutes depending on your internet connection.")
    
    # Create directory if it doesn't exist
    os.makedirs("./stl10", exist_ok=True)
    
    # Download unlabeled split
    print("\nDownloading unlabeled split (100,000 images)...")
    STL10(root="./stl10", split="unlabeled", download=True)
    
    # Download train split
    print("\nDownloading train split (5,000 images)...")
    STL10(root="./stl10", split="train", download=True)
    
    # Download test split
    print("\nDownloading test split (8,000 images)...")
    STL10(root="./stl10", split="test", download=True)
    
    print("\nDataset download complete!")
    print("The dataset is stored in ./stl10 directory")
    print("You can now proceed with training by running: python src/train.py")

if __name__ == "__main__":
    download_stl10() 