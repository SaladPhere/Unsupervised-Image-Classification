# Unsupervised Image Classification Pipeline

This project implements an unsupervised image classification pipeline using contrastive learning (SimCLR) and clustering on the STL-10 dataset. The implementation includes automatic checkpointing, experiment tracking with Weights & Biases, and comprehensive evaluation metrics.

## Project Structure

```
.
├── dataset/
│   ├── download_dataset.py  # Pre download the STL-10 dataset if you want
│   └── load_dataset.py      # STL-10 dataset loading and preprocessing
├── src/
│   ├── models/
│   │   └── simclr.py        # SimCLR model implementation
│   ├── train.py             # Training script for SimCLR
│   ├── feature_extraction.py # Feature extraction from trained model
│   ├── clustering.py        # PCA and K-means clustering
│   └── inference.py         # Inference script for new images
├── checkpoints/             # Saved model checkpoints
├── features/               # Extracted features and centroids
├── wandb/                  # Local wandb logs (when offline)
└── requirements.txt        # Project dependencies
```

## Setup

1. Create a conda environment:

```bash
conda create -n unsupervised-cls python=3.8
conda activate unsupervised-cls
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Set up Weights & Biases:

```bash
wandb login
```

## Usage

1. Train the SimCLR model:

```bash
python src/train.py
```

- Automatically saves checkpoints every 4 epochs
- Logs metrics to wandb (works offline)
- Can resume from latest checkpoint if interrupted

2. Extract features from the trained model:

```bash
python src/feature_extraction.py
```

3. Run clustering and evaluation:

```bash
python src/clustering.py
```

4. Make predictions on new images:

```bash
python src/inference.py
```

5. Visualize training and clustering results:

```bash
python visualize_features.py
```

## Implementation Details

### 1. Contrastive Learning (SimCLR)

- ResNet-50 backbone with frozen batch normalization
- 2-layer projection head (2048 → 512 → 128)
- NT-Xent loss with temperature τ = 0.2
- Mixed precision training with AdamW optimizer
- 24 epochs of training (configurable)
- Automatic checkpointing every 4 epochs
- Gradient scaling for stable training

### 2. Feature Extraction

- Extract 2048-dimensional features from the frozen backbone
- Save features for unlabeled, train, and test splits
- Efficient batch processing

### 3. Dimensionality Reduction

- PCA with 100 components (retaining >95% variance)
- Applied to all feature sets
- Explained variance tracking

### 4. Clustering

- K-means clustering with K=10 (matching STL-10 classes)
- Multi-threaded implementation for faster processing
- Hungarian matching for cluster-to-class mapping

### 5. Evaluation Metrics

- Silhouette score
- Davies-Bouldin index
- Normalized Mutual Information (NMI)
- Adjusted Rand Index (ARI)
- Hungarian accuracy

## Experiment Tracking

The project uses Weights & Biases (wandb) for experiment tracking:

### Training Metrics

- Training loss (every 100 batches)
- Epoch loss
- Learning rate
- GPU utilization
- Training progress

### Clustering Metrics

- PCA explained variance
- Clustering quality metrics
- Classification-style metrics
- Per-split performance

### Offline Support

- Metrics are stored locally in `wandb/` directory when offline
- Can be synced to wandb server later using `wandb sync`
- View metrics locally using wandb offline viewer

## Checkpoint System

- Checkpoints are saved every 4 epochs
- Each checkpoint contains:
  - Model state
  - Optimizer state
  - Current epoch
  - Training loss
  - Configuration
- Automatic resumption from latest checkpoint
- Checkpoints stored in `checkpoints/` directory
