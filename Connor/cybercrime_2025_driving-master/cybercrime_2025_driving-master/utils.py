import numpy as np
import torch
import random
import scipy.stats as stats

from torch.utils.data import Dataset, DataLoader

from models.ResNet import ResNet1D
from models.OneClassSVM import OCSVM

def id_ood_val_data(cfg, data):
    id_data = data[data['Class'] == cfg.id_class]
    ood_data = data[data['Class'] == cfg.ood_class]

    if id_data.shape[0] < cfg.window_size :
        raise ValueError(f"Not enough data in Class {cfg.id_class} for the specified window size.")
    elif ood_data.shape[0] < cfg.window_size:
        raise ValueError(f"Not enough data in Class {cfg.ood_class} for the specified window size.")

    # Drop non-variable columns
    columns_to_remove = ["Time(s)", "Class", "PathOrder"]
    id_data = id_data.drop(columns=columns_to_remove)
    ood_data = ood_data.drop(columns=columns_to_remove)

    # Split into sliding windows
    window_len = cfg.window_size
    window_overlap = cfg.window_overlap
    id_windows = create_sliding_windows(id_data, window_size=window_len, overlap=window_overlap)
    ood_windows = create_sliding_windows(ood_data, window_size=window_len, overlap=window_overlap)
    
    # Split into train, validation, and test sets
    n_id_samples = len(id_windows)
    train_idx = int(n_id_samples * cfg.train_ratio)
    val_idx = train_idx + int(n_id_samples * cfg.val_ratio)
    
    # Should the windows be shuffled?
    if cfg.shuffle_windows:
        random.Random(42).shuffle(id_windows)
    train_data = id_windows[:train_idx]
    val_data = id_windows[train_idx:val_idx]
    test_id_data = id_windows[val_idx:]

    if cfg.model.is_DL:
        return train_data, val_data, test_id_data, ood_windows
    
    # No need for validation data for ML models
    else:
        train_data += val_data
        
        # Calculate statistical features for each window
        if cfg.model.use_features:
            train_data = np.array([extract_features_from_window(window) for window in train_data])
            test_id_data = np.array([extract_features_from_window(window) for window in test_id_data])
            ood_windows = np.array([extract_features_from_window(window) for window in ood_windows])
        
        return train_data, test_id_data, ood_windows

def id_ood_val_loaders(cfg, data):
    train_data, val_data, test_id_data, ood_windows = id_ood_val_data(cfg, data)
    # Convert to dataloaders
    if cfg.model.contrastive:
        train_dataset = ContrastiveWindowDataset(train_data)
        val_dataset = ContrastiveWindowDataset(val_data)
        test_id_dataset = ContrastiveWindowDataset(test_id_data)
        ood_dataset = ContrastiveWindowDataset(ood_windows)
    else:
        train_dataset = WindowDataset(train_data)
        val_dataset = WindowDataset(val_data)
        test_id_dataset = WindowDataset(test_id_data)
        ood_dataset = WindowDataset(ood_windows)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_id_loader = DataLoader(test_id_dataset, batch_size=cfg.batch_size, shuffle=False)
    ood_loader = DataLoader(ood_dataset, batch_size=cfg.batch_size, shuffle=False)

    return train_loader, val_loader, test_id_loader, ood_loader

def extract_features_from_window(df):
    """Extract relevant features from a single time window DataFrame"""
    features = []
    
    # Statistical features for each column
    for column in df.columns:
        # Skip non-numeric or irrelevant columns
        if df[column].dtype in [np.float64, np.int64]:
            features.append(df[column].mean())
            features.append(df[column].std())
            features.append(df[column].min())
            features.append(df[column].max())
            features.append(df[column].skew())
            features.append(df[column].kurt())
            features.append(np.ptp(df[column]))
            features.append(stats.median_abs_deviation(df[column]))
            features.append(np.mean(np.abs(df[column] - 1)))         
    
    return np.array(features)


def create_sliding_windows(data, window_size=60, overlap=40):
    """
    Create sliding windows from the input data.
    
    Args:
        data: pandas DataFrame where each row is a 1-second sample
        window_size: size of each window in seconds
        overlap: overlap between consecutive windows in seconds
    
    Returns:
        List of pandas DataFrames, each representing a window
    """
    stride = window_size - overlap
    n_samples = len(data)
    windows = []
    
    # Start indices for each window
    start_indices = range(0, n_samples - window_size + 1, stride)
    
    for start_idx in start_indices:
        end_idx = start_idx + window_size
        window = data.iloc[start_idx:end_idx].copy()
        windows.append(window)
    
    return windows

class WindowDataset(Dataset):
    """Dataset for sliding windows of time series data."""
    
    def __init__(self, windows_list, transform=None):
        """
        Args:
            windows_list: List of pandas DataFrames, each representing a window
            transform: Optional transform to be applied on each window
        """
        self.windows = windows_list
        self.transform = transform
        
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = self.windows[idx]
        # Convert DataFrame to numpy array
        X = window.values.astype(np.float32)
        
        # Apply transforms if any
        if self.transform:
            X = self.transform(X)
            
        # Convert to tensor
        X = torch.tensor(X, dtype=torch.float32)
        
        return X

def init_model(cfg):
    if cfg.model.name == "ResNet":
        model = ResNet1D(cfg)
    elif cfg.model.name == "OneClassSVM":
        model = OCSVM(cfg)
    else:
        raise NotImplementedError(f"Model {cfg.model.name} not implemented.")
    return model

def init_optimiser(model, cfg):
    if cfg.model.optimiser == "Adam":
        optimiser = torch.optim.Adam(model.parameters(), lr=cfg.model.lr)
    else:
        raise NotImplementedError(f"Optimiser {cfg.model.optimiser} not implemented.")
    
    return optimiser


class ContrastiveWindowDataset(Dataset):
    def __init__(self, windows_list):
        self.windows = windows_list
        
    def __len__(self):
        return len(self.windows)
    
    def _augment(self, window):
        """Apply random augmentations to create a positive pair"""
        # Convert DataFrame to numpy array
        window_array = window.values.astype(np.float32)
        
        # Apply a combination of augmentations:
        # 1. Small Gaussian noise
        noise_level = np.random.uniform(0.01, 0.05)
        noise = np.random.normal(0, noise_level, window_array.shape)
        augmented = window_array + noise
        
        # 2. Small time shifts (up to 10% of the window)
        shift = np.random.randint(-6, 7)  # ±10% of 60s
        if shift > 0:
            augmented[shift:] = window_array[:-shift]
            augmented[:shift] = window_array[:shift]
        elif shift < 0:
            augmented[:shift] = window_array[-shift:]
            augmented[shift:] = window_array[:-shift]
            
        # 3. Random scaling (multiply by a factor between 0.9 and 1.1)
        scale_factor = np.random.uniform(0.9, 1.1)
        augmented = augmented * scale_factor
        
        return augmented
    
    def __getitem__(self, idx):
        window = self.windows[idx]
        
        # Original window
        x_i = window.values.astype(np.float32)
        
        # Augmented window (positive pair)
        x_j = self._augment(window)
        
        # Convert to tensors
        x_i = torch.tensor(x_i, dtype=torch.float32)
        x_j = torch.tensor(x_j, dtype=torch.float32)
        
        return x_i, x_j

