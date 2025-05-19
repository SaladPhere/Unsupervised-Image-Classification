import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=128):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x

class SimCLR(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature
        
        # Load pretrained ResNet-50 and freeze batch norm
        self.backbone = resnet50(weights = None)
        for name, param in self.backbone.named_parameters():
            if "bn" in name:
                param.requires_grad = False
        
        # Remove the final classification layer
        self.backbone.fc = nn.Identity()
        
        # Add projection head
        self.projection = ProjectionHead()
        
    def forward(self, x1, x2):
        # Get representations
        h1 = self.backbone(x1)
        h2 = self.backbone(x2)
        
        # Project to lower dimension
        z1 = self.projection(h1)
        z2 = self.projection(h2)
        
        return h1, h2, z1, z2
    
    def info_nce_loss(self, z1, z2):
        # Normalize projections
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Compute similarity matrix
        N = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.t()) / self.temperature
        
        # Create labels for positive pairs
        labels = torch.zeros(2 * N, 2 * N, device=z1.device)
        labels[:N, N:] = torch.eye(N, device=z1.device)
        labels[N:, :N] = torch.eye(N, device=z1.device)
        
        # Remove diagonal
        sim = sim - torch.eye(2 * N, device=z1.device) * 1e9
        
        # Compute loss
        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        mean_log_prob = (labels * log_prob).sum(1) / labels.sum(1)
        loss = -mean_log_prob.mean()
        
        return loss 