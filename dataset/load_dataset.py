from torchvision import transforms
from torchvision.datasets import STL10
from torch.utils.data import DataLoader
import torch

class SimCLRDataTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        # image is a PIL Image
        xi = self.transform(image)
        xj = self.transform(image)
        return xi, xj

def simclr_collate_fn(batch):
    # batch is a list of (xi, xj) tuples
    # Separate the xi and xj tensors
    xi_batch = torch.stack([item[0][0] for item in batch])  # Get first element of each tuple
    xj_batch = torch.stack([item[0][1] for item in batch])  # Get second element of each tuple
    return xi_batch, xj_batch

def get_data_loaders(batch_size=256):
    # Define transforms
    simclr_transforms = transforms.Compose([
        transforms.RandomResizedCrop(96),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4467, 0.4398, 0.4066],
                           std=[0.2241, 0.2215, 0.2239]),
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize(96),
        transforms.CenterCrop(96),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4467, 0.4398, 0.4066],
                           std=[0.2241, 0.2215, 0.2239]),
    ])

    # Unlabeled split for contrastive pretraining
    unlabeled_ds = STL10(
        root="./stl10", 
        split="unlabeled", 
        download=True,
        transform=SimCLRDataTransform(simclr_transforms)  # Use the wrapper
    )
    unlabeled_loader = DataLoader(
        unlabeled_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True,
        collate_fn=simclr_collate_fn  # Use the custom collate function
    )

    # Labeled train / test for evaluation
    train_ds = STL10(
        root="./stl10", 
        split="train", 
        download=True,
        transform=eval_transforms
    )
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    test_ds = STL10(
        root="./stl10", 
        split="test", 
        download=True,
        transform=eval_transforms
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return unlabeled_loader, train_loader, test_loader
