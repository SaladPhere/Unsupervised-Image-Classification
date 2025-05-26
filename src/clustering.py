import os
import sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
import wandb
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler, normalize
from joblib import parallel_backend

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.load_dataset import get_data_loaders

def compute_cluster_metrics(features, labels, true_labels=None):
    metrics = {}
    
    # Clustering metrics
    metrics['silhouette'] = silhouette_score(features, labels)
    metrics['davies_bouldin'] = davies_bouldin_score(features, labels)
    
    if true_labels is not None:
        # Clustering vs true label metrics
        metrics['nmi'] = normalized_mutual_info_score(true_labels, labels)
        metrics['ari'] = adjusted_rand_score(true_labels, labels)
        
        # Hungarian matching for accuracy
        n_clusters = max(len(np.unique(labels)), len(np.unique(true_labels)))
        confusion_matrix = np.zeros((n_clusters, n_clusters))
        for i in range(len(labels)):
            confusion_matrix[labels[i]][true_labels[i]] += 1
        
        row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
        accuracy = confusion_matrix[row_ind, col_ind].sum() / len(labels)
        metrics['hungarian_accuracy'] = accuracy
    
    return metrics

def main():
    # Load features
    print("Loading features...")
    unlabeled_features = np.load('features/unlabeled_features.npy')
    train_features = np.load('features/train_features.npy')
    test_features = np.load('features/test_features.npy')
    
    # Get data loaders for labels
    _, train_loader, test_loader = get_data_loaders(batch_size=128)
    
    # Load labels for evaluation
    train_labels = np.array([label for _, label in train_loader.dataset])
    test_labels = np.array([label for _, label in test_loader.dataset])
    
    # Initialize wandb
    wandb.init(project="unsupervised-image-classification", name="clustering")
    
    # PCA
    print("Running PCA...")
    pca = PCA(n_components=100, svd_solver="randomized")
    unlabeled_reduced = pca.fit_transform(unlabeled_features)
    train_reduced = pca.transform(train_features)
    test_reduced = pca.transform(test_features)

    scaler = StandardScaler()
    unlabeled_scaled = scaler.fit_transform(unlabeled_reduced)

    train_scaled = scaler.transform(train_reduced)
    test_scaled  = scaler.transform(test_reduced)

    unlabeled_scaled = normalize(unlabeled_scaled)
    train_scaled     = normalize(train_scaled)
    test_scaled     = normalize(test_scaled)
    
    # Log explained variance
    explained_variance = np.sum(pca.explained_variance_ratio_)
    wandb.log({'pca/explained_variance': explained_variance})
    
    # K-means clustering with parallel processing
    print("Running K-means...")
    with parallel_backend('threading', n_jobs=-1):  # Use all available CPU cores
        kmeans = KMeans(
            n_clusters=10,
            init="k-means++",
            max_iter=500,
            n_init=10,
            random_state=42,
            verbose=1
        )
        
        # Fit on unlabeled data
        unlabeled_clusters = kmeans.fit_predict(unlabeled_reduced)
        
        # Predict on train and test
        train_clusters = kmeans.predict(train_reduced)
        test_clusters = kmeans.predict(test_reduced)
    
    # Save centroids
    np.save('features/centroids.npy', kmeans.cluster_centers_)
    
    # Compute metrics
    print("Computing metrics...")
    unlabeled_metrics = compute_cluster_metrics(unlabeled_reduced, unlabeled_clusters)
    train_metrics = compute_cluster_metrics(train_reduced, train_clusters, train_labels)
    test_metrics = compute_cluster_metrics(test_reduced, test_clusters, test_labels)
    
    # Log metrics
    for split, metrics in [('unlabeled', unlabeled_metrics), 
                         ('train', train_metrics),
                         ('test', test_metrics)]:
        for metric_name, value in metrics.items():
            wandb.log({f'{split}/{metric_name}': value})
    
    wandb.finish()
    print("Clustering complete!")

if __name__ == "__main__":
    main() 