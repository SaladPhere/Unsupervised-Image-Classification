import torch
import numpy as np
from torchvision import transforms
from models.simclr import SimCLR
from PIL import Image
from sklearn.decomposition import PCA

class UnsupervisedClassifier:
    def __init__(self, model_path, centroids_path, pca_components=100):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        checkpoint = torch.load(model_path)
        self.model = SimCLR().to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load centroids
        self.centroids = np.load(centroids_path)
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize(96),
            transforms.CenterCrop(96),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4467, 0.4398, 0.4066],
                              std=[0.2241, 0.2215, 0.2239])
        ])
        
        # Load PCA components
        self.pca = PCA(n_components=pca_components)
        self.pca.fit(np.load('features/unlabeled_features.npy'))
    
    def predict(self, image_path):
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model.backbone(image)
            features = features.cpu().numpy()
        
        # Apply PCA
        features_reduced = self.pca.transform(features)
        
        # Find nearest centroid
        distances = np.linalg.norm(self.centroids - features_reduced, axis=1)
        cluster_id = np.argmin(distances)
        
        return cluster_id

def main():
    # Example usage
    classifier = UnsupervisedClassifier(
        model_path='checkpoints/simclr_epoch_200.pt',
        centroids_path='features/centroids.npy'
    )
    
    # Example prediction
    image_path = 'path/to/your/image.jpg'
    cluster_id = classifier.predict(image_path)
    print(f"Predicted cluster: {cluster_id}")

if __name__ == "__main__":
    main() 