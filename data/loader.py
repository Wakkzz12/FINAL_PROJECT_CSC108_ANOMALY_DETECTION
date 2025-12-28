import pandas as pd
import numpy as np
from sklearn.utils import resample

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================
# The Credit Card Fraud dataset is highly imbalanced (99.83% legitimate)
# This module handles loading and optional balancing for better learning
# =============================================================================

def load_data(path, balance=False, max_samples=None):
    """
    Load and preprocess credit card transaction data.
    
    Args:
        path (str): Path to creditcard.csv file
        balance (bool): Whether to balance classes
        max_samples (int): Maximum samples to use per class (None = use all fraud)
        
    Returns:
        pd.DataFrame: Processed transaction data
    """
    print(f"Loading data from {path}...")
    df = pd.read_csv(f"data/{path}")
    
    print(f"Original dataset size: {len(df)} transactions")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
    
    # Handle missing values
    if df.isnull().sum().sum() > 0:
        print("Warning: Missing values detected. Filling with column means.")
        df = df.fillna(df.mean())
    
    # Balance the dataset if requested
    if balance:
        # Calculate samples per class
        if max_samples is not None:
            samples_per_class = max_samples // 2  # Half for fraud, half for legit
        else:
            samples_per_class = None  # Use all available fraud
        
        df = balance_dataset(df, target_size=samples_per_class)
    elif max_samples is not None:
        # Just limit total size without balancing
        print(f"Limiting to {max_samples} samples...")
        df = df.sample(n=min(len(df), max_samples), random_state=42).reset_index(drop=True)
    
    print(f"Final dataset size: {len(df)} transactions")
    print(f"Final fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
    print("-" * 50)
    
    return df


def balance_dataset(df, target_size=None):
    """
    Balance dataset using combination of undersampling and oversampling.
    
    Args:
        df (pd.DataFrame): Original imbalanced dataset
        target_size (int): Desired size per class (None = match minority class)
        
    Returns:
        pd.DataFrame: Balanced dataset
    """
    print("Balancing dataset...")
    
    # Separate by class
    df_fraud = df[df['Class'] == 1]
    df_legit = df[df['Class'] == 0]
    
    print(f"  Fraud transactions: {len(df_fraud)}")
    print(f"  Legitimate transactions: {len(df_legit)}")
    
    # If target_size specified, oversample fraud to reach it
    if target_size is not None:
        samples_per_class = target_size
        
        # Oversample minority class (fraud) with replacement
        if len(df_fraud) < samples_per_class:
            df_fraud_sampled = resample(
                df_fraud,
                replace=True,  # Allow duplicates
                n_samples=samples_per_class,
                random_state=42
            )
        else:
            df_fraud_sampled = df_fraud.sample(n=samples_per_class, random_state=42)
        
        # Undersample majority class (legitimate)
        df_legit_sampled = resample(
            df_legit,
            replace=False,
            n_samples=samples_per_class,
            random_state=42
        )
    else:
        # Default: match minority class size
        df_fraud_sampled = df_fraud
        df_legit_sampled = resample(
            df_legit,
            replace=False,
            n_samples=len(df_fraud),
            random_state=42
        )
    
    # Combine and shuffle
    df_balanced = pd.concat([df_fraud_sampled, df_legit_sampled])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"  Balanced dataset: {len(df_balanced)} transactions ({len(df_fraud_sampled)} fraud + {len(df_legit_sampled)} legit)")
    
    return df_balanced


def split_train_test(df, train_ratio=0.8):
    """
    Split data into training and testing sets.
    
    Time Complexity: O(N)
    Space Complexity: O(N)
    
    Args:
        df (pd.DataFrame): Full dataset
        train_ratio (float): Proportion for training (0 to 1)
        
    Returns:
        tuple: (train_df, test_df)
    """
    split_idx = int(len(df) * train_ratio)
    
    # Sequential split (maintains temporal order if important)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    
    print(f"Training set: {len(train_df)} transactions")
    print(f"Testing set: {len(test_df)} transactions")
    
    return train_df, test_df