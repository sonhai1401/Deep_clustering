import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# 1. Custom Attention Modules
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels//ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

# 2. Modified ResNet50 with Attention
class AttentiveResNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Initial convolution block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks with attention
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        
        # Attention modules
        self.ca1 = ChannelAttention(64)
        self.sa1 = SpatialAttention()
        self.ca2 = ChannelAttention(128)
        self.sa2 = SpatialAttention()
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        
        # Layer 1 with attention
        x = self.layer1(x)
        x = self.ca1(x) * x
        x = self.sa1(x) * x
        
        # Layer 2 with attention
        x = self.layer2(x)
        x = self.ca2(x) * x
        x = self.sa2(x) * x
        
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = F.adaptive_avg_pool2d(x, (1,1)).squeeze()
        return self.projection(x)

# 3. Hybrid Contrastive Loss
class ClusterContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, alpha=0.5):
        super().__init__()
        self.temp = temperature
        self.alpha = alpha
        
    def forward(self, features, prototypes, pseudo_labels):
        # Instance-level contrast
        sim_matrix = torch.mm(features, features.t()) / self.temp
        pos_mask = (pseudo_labels.unsqueeze(1) == pseudo_labels.unsqueeze(0)).float()
        neg_mask = 1 - pos_mask
        
        # Cluster-level contrast
        proto_sim = torch.mm(features, prototypes.t()) / self.temp
        
        # Loss components
        instance_loss = -torch.log((sim_matrix.exp() * pos_mask).sum(1) / sim_matrix.exp().sum(1)).mean()
        cluster_loss = -torch.log_softmax(proto_sim, dim=1).mean()
        
        return self.alpha * instance_loss + (1 - self.alpha) * cluster_loss

# 4. Clustering Module
class DeepCluster(nn.Module):
    def __init__(self, num_clusters=10):
        super().__init__()
        self.backbone = AttentiveResNet()
        self.prototypes = nn.Parameter(torch.randn(num_clusters, 128))
        
    def forward(self, x):
        features = self.backbone(x)
        return F.normalize(features, dim=1)
    
    def predict(self, x):
        with torch.no_grad():
            features = self.forward(x)
            return torch.argmax(torch.mm(features, self.prototypes.t()), dim=1)

# 5. Training Pipeline
def train(model, dataloader, num_epochs=100):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    criterion = ClusterContrastiveLoss()
    
    for epoch in range(num_epochs):
        # Step 1: Generate pseudo-labels
        all_features = []
        for images, _ in dataloader:
            features = model(images.to(device))
            all_features.append(features.detach())
        all_features = torch.cat(all_features)
        
        # K-means clustering
        cluster_ids, _ = kmeans_torch(all_features, num_clusters=10)
        cluster_ids = torch.tensor(cluster_ids).to(device)
        
        # Step 2: Contrastive training
        total_loss = 0.0
        for (images, _), batch_ids in zip(dataloader, cluster_ids.split(512)):
            images = images.to(device)
            
            features = model(images)
            loss = criterion(features, model.prototypes, batch_ids)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs} Loss: {total_loss/len(dataloader):.4f}')

# 6. Evaluation
def evaluate(model, dataloader):
    true_labels, pred_labels = [], []
    for images, labels in dataloader:
        preds = model.predict(images.to(device)).cpu()
        true_labels.append(labels)
        pred_labels.append(preds)
    
    true_labels = torch.cat(true_labels).numpy()
    pred_labels = torch.cat(pred_labels).numpy()
    
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    return nmi, ari

def kmeans_torch(X, num_clusters, num_iters=10):
    """
    Simple KMeans implementation using PyTorch.
    Args:
        X: torch.Tensor of shape (N, D) - input features
        num_clusters: number of clusters K
        num_iters: number of iterations
    Returns:
        cluster_ids: torch.LongTensor of shape (N,)
        centroids: torch.Tensor of shape (K, D)
    """
    N, D = X.shape

    # Randomly initialize centroids from data points
    indices = torch.randperm(N)[:num_clusters]
    centroids = X[indices]

    for _ in range(num_iters):
        # Compute pairwise distances (N x K)
        distances = torch.cdist(X, centroids, p=2)  # Euclidean distance
        cluster_ids = distances.argmin(dim=1)

        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        for k in range(num_clusters):
            if (cluster_ids == k).sum() == 0:
                new_centroids[k] = X[torch.randint(0, N, (1,))]  # Random if no points
            else:
                new_centroids[k] = X[cluster_ids == k].mean(dim=0)
        centroids = new_centroids

    return cluster_ids, centroids

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Thêm transform chuẩn hóa
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # CIFAR-10
])

# Initialize and train
model = DeepCluster().to(device)
train_loader = DataLoader(CIFAR10('/kaggle/input/cifar10/cifar-10-python', transform=transform), batch_size=512, shuffle=True)
train(model, train_loader)

# Evaluate
test_loader = DataLoader(CIFAR10('/kaggle/input/cifar10/cifar-10-python', train=False, transform=transform), batch_size=512)
nmi, ari = evaluate(model, test_loader)
print(f'NMI: {nmi:.4f}, ARI: {ari:.4f}')
