import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

# 1D ResNet model
class ResNet1D(nn.Module):
    def __init__(self, cfg):
        super(ResNet1D, self).__init__()
        self.in_channels = cfg.model.in_channels
        self.layers = cfg.model.layers
        self.block = BasicBlock1D
        self.projection_dim = cfg.model.projection_dim
        self.device = cfg.device
        self.num_epochs = cfg.num_epochs

        self.loss_function = NTXentLoss(cfg.model.temperature)
        
        # Initial layers
        self.conv1 = nn.Conv1d(cfg.num_vars, 64, kernel_size=cfg.model.kernel_size_1, stride=cfg.model.stride_1, padding=cfg.model.padding_1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=cfg.model.kernel_size_2, stride=cfg.model.stride_2, padding=cfg.model.padding_2)
        
        # Residual blocks
        self.layer1 = self._make_layer(self.block, 64, self.layers[0])
        self.layer2 = self._make_layer(self.block, 128,self.layers[1], stride=2)
        self.layer3 = self._make_layer(self.block, 256,self.layers[2], stride=2)
        self.layer4 = self._make_layer(self.block, 512,self.layers[3], stride=2)
        
        # Global average pooling and projection head
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * self.block.expansion, 512)
        
        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.projection_dim)
        )
        
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward_features(self, x):
        # x shape: [batch_size, sequence_length, features]
        # Convert to [batch_size, features, sequence_length] for 1D convolution
        x = x.transpose(1, 2)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def forward(self, x):
        features = self.forward_features(x)
        projections = self.projection(features)
        return F.normalize(projections, dim=1)  # L2 normalize for cosine similarity

    def compute_loss(self, z_i, z_j):
        return self.loss_function(z_i, z_j)
    
    def train_model(self, train_loader, val_loader, optimiser, cfg):
        
        self.to(self.device)      
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            # Training
            self.train()
            train_loss = 0.0
            
            for x_i, x_j in train_loader:
                x_i, x_j = x_i.to(self.device), x_j.to(self.device)
                
                # Forward pass
                z_i = self.forward(x_i)
                z_j = self.forward(x_j)
                
                # Compute loss
                loss = self.compute_loss(z_i, z_j)
                
                # Backward pass
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            self.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for x_i, x_j in val_loader:
                    x_i, x_j = x_i.to(self.device), x_j.to(self.device)
                    
                    z_i = self.forward(x_i)
                    z_j = self.forward(x_j)
                    
                    loss = self.compute_loss(z_i, z_j)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Save the best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.state_dict(), "best_contrastive_model.pt")
        
        # Load the best model
        self.load_state_dict(torch.load("best_contrastive_model.pt"))
        print("Training completed.")

    def compute_ood_score(self, x_i):
        z_i = self.forward(x_i)
        sim = F.cosine_similarity(z_i.unsqueeze(1), z_i.unsqueeze(0), dim=2)
        # DEBUG
        if torch.sum(torch.isnan(sim)) > 0 or torch.sum(torch.isinf(sim)) > 0:
            print("Nans in similarity matrix")
            print(sim)
        return sim
    
    def test_model(self, test_id_loader, ood_loader):
        self.to(self.device)
        self.eval()
        sim_scores_id, sim_scores_ood = [], []

        with torch.no_grad():
            for x_i, x_j in test_id_loader:
                x_i = x_i.to(self.device)
                sim = self.compute_ood_score(x_i)
                batch_scores = (torch.sum(sim, dim=1) - 1) / (sim.size(0) - 1 + 1e-8)
                sim_scores_id.extend(batch_scores.cpu().numpy())

        with torch.no_grad():
            for x_i, x_j in ood_loader:
                x_i = x_i.to(self.device)
                sim = self.compute_ood_score(x_i)
                batch_scores = (torch.sum(sim, dim=1) - 1) / (sim.size(0) - 1 + 1e-8)
                sim_scores_ood.extend(batch_scores.cpu().numpy())

        ood_labels = [0]*len(sim_scores_id)+[1]*len(sim_scores_ood)
        sim_scores = np.array(sim_scores_id + sim_scores_ood) # higher scores => more likely OOD

        return sim_scores, ood_labels




# Basic 1D ResNet block
class BasicBlock1D(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


# Create 1D ResNet-18 model
def resnet18_1d(num_features, projection_dim=128):
    return ResNet1D(BasicBlock1D, [2, 2, 2, 2], num_features, projection_dim)


# Contrastive loss (NT-Xent loss)
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, z_i, z_j):
        # Concatenate the representations for pairs
        batch_size = z_i.size(0)
        representations = torch.cat([z_i, z_j], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), 
                                              representations.unsqueeze(0), 
                                              dim=2)
        
        # Remove diagonal elements (self-similarity)
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        # Remove the positives from the similarity matrix
        mask = (~torch.eye(batch_size * 2, dtype=bool, device=z_i.device)).float()
        similarity_matrix = similarity_matrix * mask
        
        # For numerical stability
        numerator = torch.exp(positives / self.temperature)
        denominator = torch.sum(torch.exp(similarity_matrix / self.temperature), dim=1)
        
        losses = -torch.log(numerator / denominator)
        return torch.mean(losses)

