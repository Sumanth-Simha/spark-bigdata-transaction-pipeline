import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pickle
import os

class FraudDataset(Dataset):
    def __init__(self, numerical_features, categorical_features, labels=None):
        self.numerical_features = torch.tensor(numerical_features, dtype=torch.float32)
        self.categorical_features = torch.tensor(categorical_features, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None else None

    def __len__(self):
        return len(self.numerical_features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.numerical_features[idx], self.categorical_features[idx], self.labels[idx]
        return self.numerical_features[idx], self.categorical_features[idx]

def preprocess_data(df, is_training=True, scaler_path="src/models/scaler.pkl"):
    """
    Preprocess the dataframe for the neural network.
    Extracts 'Hour' from 'Time', standardizes numerical features.
    """
    df = df.copy()

    # Feature Engineering: Extract 'Hour' from 'Time' (assuming time is in seconds)
    if 'Time' in df.columns:
        df['Hour'] = (df['Time'] // 3600) % 24
    else:
        df['Hour'] = 0 # Fallback if no time column is present

    # Define categorical and numerical columns
    categorical_cols = ['Hour']

    # Exclude 'Class' if present, and 'Hour', keep the rest as numerical
    exclude_cols = ['Class', 'Hour']
    numerical_cols = [col for col in df.columns if col not in exclude_cols]

    if is_training:
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

        # Save the scaler for inference
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
    else:
        # Load the scaler
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            df[numerical_cols] = scaler.transform(df[numerical_cols])
        else:
            print(f"Warning: Scaler not found at {scaler_path}. Applying fit_transform instead.")
            scaler = StandardScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Extract numerical and categorical features
    numerical_features = df[numerical_cols].values
    categorical_features = df[categorical_cols].values

    labels = df['Class'].values if 'Class' in df.columns else None

    return numerical_features, categorical_features, labels, len(numerical_cols)

def get_dataloaders(data_path, batch_size=2048, val_split=0.2, scaler_path="src/models/scaler.pkl"):
    """
    Loads data, preprocesses it, and returns Train and Val DataLoaders.
    """
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        # Assume parquet
        df = pd.read_parquet(data_path)

    # Basic shuffling before split
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    split_idx = int(len(df) * (1 - val_split))
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    train_num, train_cat, train_labels, num_features_count = preprocess_data(train_df, is_training=True, scaler_path=scaler_path)
    val_num, val_cat, val_labels, _ = preprocess_data(val_df, is_training=False, scaler_path=scaler_path)

    train_dataset = FraudDataset(train_num, train_cat, train_labels)
    val_dataset = FraudDataset(val_num, val_cat, val_labels)

    # Use a large batch size since the dataset is large
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, num_features_count
