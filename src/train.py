import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
import wandb
from tqdm import tqdm
import numpy as np

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.simclr import SimCLR
from dataset.load_dataset import get_data_loaders

def train_simclr(
    model,
    train_loader,
    optimizer,
    scaler,
    epoch,
    device,
    temperature=0.2,
    use_wandb=True
):
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, batch_data in enumerate(pbar):
        x1, x2 = batch_data
        
        # Move data to GPU
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast('cuda'):
            h1, h2, z1, z2 = model(x1, x2)
            loss = model.info_nce_loss(z1, z2)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': total_loss / (batch_idx + 1)})
        
        if use_wandb and batch_idx % 100 == 0:
            wandb.log({
                'train/loss': loss.item(),
                'train/epoch': epoch,
                'train/step': epoch * len(train_loader) + batch_idx
            })
    
    return total_loss / len(train_loader)

def main():
    # Enable CUDA benchmarking for better performance
    torch.backends.cudnn.benchmark = True
    
    # Training configuration
    config = {
        'batch_size': 256,  # Adjust based on your GPU memory
        'temperature': 0.2,
        'learning_rate': 1.2e-3,
        'weight_decay': 1e-4,
        'num_epochs': 200
    }
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Initialize wandb
    wandb.init(
        project="unsupervised-image-classification",
        name="simclr-training",
        config=config
    )
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    else:
        print("WARNING: CUDA is not available. Using CPU instead.")
    
    # Get data loaders with specified batch size
    unlabeled_loader, train_loader, test_loader = get_data_loaders(
        batch_size=config['batch_size']
    )
    
    # Initialize model and move to GPU
    model = SimCLR(temperature=config['temperature']).to(device)
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Initialize gradient scaler for mixed precision
    scaler = torch.amp.GradScaler('cuda')
    
    # Try to load the latest checkpoint
    start_epoch = 0
    checkpoint_files = [f for f in os.listdir('checkpoints') if f.startswith('simclr_epoch_')]
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        checkpoint = torch.load(f'checkpoints/{latest_checkpoint}', map_location=device, weights_only= True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    for epoch in range(start_epoch, config['num_epochs']):
        train_loss = train_simclr(
            model=model,
            train_loader=unlabeled_loader,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            device=device,
            temperature=config['temperature']
        )
        
        # Save checkpoint
        if (epoch + 1) % 4 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'config': config
            }
            torch.save(checkpoint, f'checkpoints/simclr_epoch_{epoch+1}.pt')
            
        # Log epoch metrics
        wandb.log({
            'epoch': epoch,
            'train/epoch_loss': train_loss
        })
    
    wandb.finish()

if __name__ == "__main__":
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    main() 