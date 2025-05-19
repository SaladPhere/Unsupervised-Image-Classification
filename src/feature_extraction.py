import os
import sys
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.simclr import SimCLR
from dataset.load_dataset import get_data_loaders

def extract_features(model, dataloader, device):
    features = []
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            # Handle different data formats
            if isinstance(batch, tuple):
                # For unlabeled data (paired views)
                x = batch[0]  # Use first view only
            elif isinstance(batch, list):
                # For single images
                x = batch[0]
            else:
                x = batch
            
            # Ensure x is a tensor
            if not isinstance(x, torch.Tensor):
                x = torch.stack(x)
            
            x = x.to(device)
            
            # Get backbone features
            h = model.backbone(x)
            features.append(h.cpu().numpy())
    
    return np.concatenate(features, axis=0)

def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get data loaders
    unlabeled_loader, train_loader, test_loader = get_data_loaders(batch_size=128)
    
    # Find latest checkpoint
    checkpoint_dir = 'checkpoints'
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('simclr_epoch_') and f.endswith('.pt')]
    if not checkpoints:
        raise FileNotFoundError("No checkpoints found in checkpoints directory")
    
    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load trained model
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model = SimCLR().to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}")
    
    # Create features directory
    os.makedirs('features', exist_ok=True)
    
    # Extract features for all splits
    print("Extracting features from unlabeled data...")
    unlabeled_features = extract_features(model, unlabeled_loader, device)
    np.save('features/unlabeled_features.npy', unlabeled_features)
    print(f"Unlabeled features shape: {unlabeled_features.shape}")
    
    print("Extracting features from train data...")
    train_features = extract_features(model, train_loader, device)
    np.save('features/train_features.npy', train_features)
    print(f"Train features shape: {train_features.shape}")
    
    print("Extracting features from test data...")
    test_features = extract_features(model, test_loader, device)
    np.save('features/test_features.npy', test_features)
    print(f"Test features shape: {test_features.shape}")
    
    print("Feature extraction complete!")

if __name__ == "__main__":
    main() 