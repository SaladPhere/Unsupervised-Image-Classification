import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize
import torch
import sys
import os
from joblib import parallel_backend

# Add src to path for dataset loading
sys.path.append('src')
sys.path.append('dataset')

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with: pip install umap-learn")

from dataset.load_dataset import get_data_loaders
# Import the clustering function to ensure consistency
from src.clustering import compute_cluster_metrics

# STL-10 class names for better visualization
STL10_CLASSES = ['airplane', 'bird', 'car', 'cat', 'deer', 
                 'dog', 'horse', 'monkey', 'ship', 'truck']

def load_data():
    """Load all features and labels - same as clustering.py."""
    print("Loading features and labels...")
    
    # Load features
    unlabeled_features = np.load('features/unlabeled_features.npy')
    train_features = np.load('features/train_features.npy')
    test_features = np.load('features/test_features.npy')
    centroids = np.load('features/centroids.npy')
    
    # Get data loaders for labels - same as clustering.py
    _, train_loader, test_loader = get_data_loaders(batch_size=128)
    
    # Load labels for evaluation - same as clustering.py
    train_labels = np.array([label for _, label in train_loader.dataset])
    test_labels = np.array([label for _, label in test_loader.dataset])
    
    print(f"Unlabeled features: {unlabeled_features.shape}")
    print(f"Train features: {train_features.shape}")
    print(f"Test features: {test_features.shape}")
    print(f"Centroids: {centroids.shape}")
    
    return (unlabeled_features, train_features, test_features, centroids,
            train_labels, test_labels)

def apply_exact_clustering_process(unlabeled_features, train_features, test_features):
    """Apply the exact same clustering process as clustering.py."""
    print("Applying exact clustering process from clustering.py...")
    
    # PCA - exact same as clustering.py
    print("Running PCA...")
    pca = PCA(n_components=100, svd_solver="randomized")
    unlabeled_reduced = pca.fit_transform(unlabeled_features)
    train_reduced = pca.transform(train_features)
    test_reduced = pca.transform(test_features)

    # Scaling and normalization - exact same as clustering.py
    scaler = StandardScaler()
    unlabeled_scaled = scaler.fit_transform(unlabeled_reduced)
    train_scaled = scaler.transform(train_reduced)
    test_scaled = scaler.transform(test_reduced)

    unlabeled_scaled = normalize(unlabeled_scaled)
    train_scaled = normalize(train_scaled)
    test_scaled = normalize(test_scaled)
    
    # K-means clustering with parallel processing - exact same as clustering.py
    print("Running K-means...")
    with parallel_backend('threading', n_jobs=-1):
        kmeans = KMeans(
            n_clusters=10,
            init="k-means++",
            max_iter=500,
            n_init=10,
            random_state=42,
            verbose=1
        )
        
        # Fit on unlabeled data
        unlabeled_clusters = kmeans.fit_predict(unlabeled_scaled)
        
        # Predict on train and test
        train_clusters = kmeans.predict(train_scaled)
        test_clusters = kmeans.predict(test_scaled)
    
    return (unlabeled_clusters, train_clusters, test_clusters,
            unlabeled_scaled, train_scaled, test_scaled,
            pca, scaler, kmeans)

def reduce_dimensions_3d(features, method='tsne', n_samples=5000, random_state=42):
    """Reduce features to 3D using t-SNE or UMAP."""
    print(f"Reducing dimensions to 3D using {method.upper()}...")
    
    # Sample data if too large
    if len(features) > n_samples:
        indices = np.random.choice(len(features), n_samples, replace=False)
        features_sample = features[indices]
        print(f"Sampling {n_samples} points from {len(features)} total points")
    else:
        features_sample = features
        indices = np.arange(len(features))
    
    if method == 'tsne':
        reducer = TSNE(n_components=3, random_state=random_state, 
                      perplexity=min(30, len(features_sample)//4))
        features_3d = reducer.fit_transform(features_sample)
    elif method == 'umap' and UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=3, random_state=random_state)
        features_3d = reducer.fit_transform(features_sample)
    elif method == 'pca':
        reducer = PCA(n_components=3, random_state=random_state)
        features_3d = reducer.fit_transform(features_sample)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return features_3d, indices

def create_3d_scatter(features_3d, labels, title, save_path, 
                     class_names=None, alpha=0.6):
    """Create a 3D scatter plot."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create colormap
    colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(labels))))
    
    for i, label in enumerate(np.unique(labels)):
        mask = labels == label
        if class_names and label < len(class_names):
            label_name = f"{label}: {class_names[label]}"
        else:
            label_name = f"Cluster {label}"
            
        ax.scatter(features_3d[mask, 0], 
                  features_3d[mask, 1], 
                  features_3d[mask, 2],
                  c=[colors[i]], 
                  label=label_name,
                  alpha=alpha,
                  s=20)
    
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")

def visualize_with_centroids(features_3d, cluster_labels, centroids_3d, title, save_path):
    """Create 3D visualization with cluster centroids."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot data points
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i in range(10):
        mask = cluster_labels == i
        if np.any(mask):
            ax.scatter(features_3d[mask, 0], 
                      features_3d[mask, 1], 
                      features_3d[mask, 2],
                      c=[colors[i]], 
                      label=f'Cluster {i}',
                      alpha=0.6,
                      s=20)
    
    # Plot centroids
    ax.scatter(centroids_3d[:, 0], 
              centroids_3d[:, 1], 
              centroids_3d[:, 2],
              c='red', 
              marker='x', 
              s=200, 
              linewidths=3,
              label='Centroids')
    
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")

def compare_clusters_vs_true_labels(features_3d, cluster_labels, true_labels, 
                                   indices, split_name):
    """Create side-by-side comparison of clusters vs true labels."""
    fig = plt.figure(figsize=(20, 8))
    
    # Plot 1: Cluster assignments
    ax1 = fig.add_subplot(121, projection='3d')
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i in range(10):
        mask = cluster_labels[indices] == i
        if np.any(mask):
            ax1.scatter(features_3d[mask, 0], 
                       features_3d[mask, 1], 
                       features_3d[mask, 2],
                       c=[colors[i]], 
                       label=f'Cluster {i}',
                       alpha=0.6,
                       s=20)
    
    ax1.set_title(f'{split_name} - Cluster Assignments')
    ax1.set_xlabel('Component 1')
    ax1.set_ylabel('Component 2')
    ax1.set_zlabel('Component 3')
    ax1.legend()
    
    # Plot 2: True labels
    ax2 = fig.add_subplot(122, projection='3d')
    
    for i in range(10):
        mask = true_labels[indices] == i
        if np.any(mask):
            ax2.scatter(features_3d[mask, 0], 
                       features_3d[mask, 1], 
                       features_3d[mask, 2],
                       c=[colors[i]], 
                       label=STL10_CLASSES[i],
                       alpha=0.6,
                       s=20)
    
    ax2.set_title(f'{split_name} - True Labels')
    ax2.set_xlabel('Component 1')
    ax2.set_ylabel('Component 2')
    ax2.set_zlabel('Component 3')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'visualizations/{split_name.lower()}_clusters_vs_true_3d.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: visualizations/{split_name.lower()}_clusters_vs_true_3d.png")

def main():
    # Create output directory
    os.makedirs('visualizations', exist_ok=True)
    
    # Load data
    (unlabeled_features, train_features, test_features, centroids,
     train_labels, test_labels) = load_data()
    
    # Apply exact clustering process from clustering.py
    (unlabeled_clusters, train_clusters, test_clusters,
     unlabeled_scaled, train_scaled, test_scaled,
     pca, scaler, kmeans) = apply_exact_clustering_process(
         unlabeled_features, train_features, test_features)
    
    # Verify our clustering matches the original results
    print("\nVerifying clustering consistency...")
    train_metrics = compute_cluster_metrics(train_scaled, train_clusters, train_labels)
    test_metrics = compute_cluster_metrics(test_scaled, test_clusters, test_labels)
    
    print(f"Train Hungarian Accuracy: {train_metrics['hungarian_accuracy']:.4f}")
    print(f"Test Hungarian Accuracy: {test_metrics['hungarian_accuracy']:.4f}")
    print("(These should match your clustering.py results)")
    
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)
    
    # 1. Visualize unlabeled data with clusters
    print("\n1. Visualizing unlabeled data clusters...")
    methods = ['pca', 'tsne']
    if UMAP_AVAILABLE:
        methods.append('umap')
    
    for method in methods:
        print(f"\nUsing {method.upper()} for dimensionality reduction...")
        
        unlabeled_3d, unlabeled_indices = reduce_dimensions_3d(
            unlabeled_scaled, method=method, n_samples=5000)
        
        create_3d_scatter(
            unlabeled_3d, 
            unlabeled_clusters[unlabeled_indices],
            f'Unlabeled Data Clusters ({method.upper()})',
            f'visualizations/unlabeled_clusters_{method}_3d.png'
        )
        
        # Visualize with centroids (only for first method to avoid redundancy)
        if method == methods[0]:
            # The centroids are already in PCA space (100 dimensions), so just apply scaler and normalize
            centroids_scaled = normalize(scaler.transform(centroids))
            
            centroids_3d, _ = reduce_dimensions_3d(
                np.vstack([unlabeled_scaled[unlabeled_indices], centroids_scaled]), 
                method=method, n_samples=len(unlabeled_indices)+10)
            
            visualize_with_centroids(
                centroids_3d[:-10], 
                unlabeled_clusters[unlabeled_indices],
                centroids_3d[-10:],
                f'Unlabeled Data with Centroids ({method.upper()})',
                f'visualizations/unlabeled_with_centroids_{method}_3d.png'
            )
    
    # 2. Visualize train set
    print("\n2. Visualizing train set...")
    train_3d, train_indices = reduce_dimensions_3d(
        train_scaled, method='tsne', n_samples=min(3000, len(train_scaled)))
    
    compare_clusters_vs_true_labels(
        train_3d, train_clusters, train_labels, train_indices, 'Train')
    
    # 3. Visualize test set
    print("\n3. Visualizing test set...")
    test_3d, test_indices = reduce_dimensions_3d(
        test_scaled, method='tsne', n_samples=min(3000, len(test_scaled)))
    
    compare_clusters_vs_true_labels(
        test_3d, test_clusters, test_labels, test_indices, 'Test')
    
    # 4. Combined visualization of all splits
    print("\n4. Creating combined visualization...")
    
    # Combine features for joint dimensionality reduction
    all_features = np.vstack([
        train_scaled[:1000],  # Sample to keep manageable
        test_scaled[:1500],
        unlabeled_scaled[:1500]
    ])
    
    all_3d, _ = reduce_dimensions_3d(all_features, method='tsne', n_samples=4000)
    
    # Create split labels
    split_labels = (['Train'] * 1000 + ['Test'] * 1500 + ['Unlabeled'] * 1500)[:len(all_3d)]
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = {'Train': 'red', 'Test': 'blue', 'Unlabeled': 'gray'}
    for split in ['Train', 'Test', 'Unlabeled']:
        mask = np.array(split_labels) == split
        if np.any(mask):
            ax.scatter(all_3d[mask, 0], 
                      all_3d[mask, 1], 
                      all_3d[mask, 2],
                      c=colors[split], 
                      label=split,
                      alpha=0.6,
                      s=20)
    
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.set_title('All Splits Combined (t-SNE)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('visualizations/all_splits_combined_3d.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: visualizations/all_splits_combined_3d.png")
    
    print("\n" + "="*50)
    print("VISUALIZATION COMPLETE!")
    print("="*50)
    print("Check the 'visualizations/' folder for all generated plots.")
    print("\nGenerated visualizations:")
    for filename in os.listdir('visualizations'):
        if filename.endswith('.png'):
            print(f"  - {filename}")

if __name__ == "__main__":
    main() 