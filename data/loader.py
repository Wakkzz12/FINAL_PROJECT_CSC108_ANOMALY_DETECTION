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
    
    Dataset features (from Kaggle Credit Card Fraud Detection):
    - Time: Seconds elapsed between this transaction and first transaction
    - V1-V28: PCA-transformed features (confidential for privacy)
    - Amount: Transaction amount
    - Class: Label (0=legitimate, 1=fraud)
    
    Time Complexity: O(N) where N = number of transactions
    Space Complexity: O(N) for storing dataframe
    
    Args:
        path (str): Path to creditcard.csv file
        balance (bool): Whether to balance classes (helps with imbalanced data)
        max_samples (int): Maximum samples to use (None = use all)
        
    Returns:
        pd.DataFrame: Processed transaction data
    """
    print(f"Loading data from {path}...")
    df = pd.read_csv(f"data/{path}")
    
    print(f"Original dataset size: {len(df)} transactions")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
    
    # Handle missing values (if any)
    # Time Complexity: O(N * M) where M = number of columns
    if df.isnull().sum().sum() > 0:
        print("Warning: Missing values detected. Filling with column means.")
        df = df.fillna(df.mean())
    
    # Optional: Balance the dataset to improve learning
    # Credit card fraud is extremely imbalanced (~0.17% fraud)
    # Balancing helps Q-Learning agent learn fraud patterns better
    if balance:
        df = balance_dataset(df)
    
    # Optional: Limit dataset size for faster experimentation
    # Time Complexity: O(1) - just slicing
    if max_samples is not None and len(df) > max_samples:
        print(f"Limiting to {max_samples} samples for faster training...")
        # Stratified sampling to maintain class distribution
        fraud = df[df['Class'] == 1]
        legit = df[df['Class'] == 0]
        
        fraud_samples = min(len(fraud), max_samples // 2)
        legit_samples = max_samples - fraud_samples
        
        df = pd.concat([
            fraud.sample(n=fraud_samples, random_state=42),
            legit.sample(n=legit_samples, random_state=42)
        ]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Final dataset size: {len(df)} transactions")
    print(f"Final fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
    print("-" * 50)
    
    return df


def balance_dataset(df):
    """
    Balance dataset using random undersampling of majority class.
    
    For fraud detection:
    - Majority class (legitimate): ~99.83%
    - Minority class (fraud): ~0.17%
    
    Undersampling reduces legitimate transactions to match fraud count.
    Alternative: Could use oversampling (SMOTE) but undersampling is simpler.
    
    Time Complexity: O(N_minority) where N_minority = fraud count
    Space Complexity: O(N_minority * 2) for balanced dataset
    
    Args:
        df (pd.DataFrame): Original imbalanced dataset
        
    Returns:
        pd.DataFrame: Balanced dataset
    """
    print("Balancing dataset...")
    
    # Separate by class
    df_fraud = df[df['Class'] == 1]
    df_legit = df[df['Class'] == 0]
    
    print(f"  Fraud transactions: {len(df_fraud)}")
    print(f"  Legitimate transactions: {len(df_legit)}")
    
    # Undersample majority class to match minority class size
    # Using sklearn.utils.resample for random sampling
    df_legit_downsampled = resample(
        df_legit,
        replace=False,  # Sample without replacement
        n_samples=len(df_fraud),  # Match minority class size
        random_state=42  # Reproducibility
    )
    
    # Combine minority class with downsampled majority class
    df_balanced = pd.concat([df_fraud, df_legit_downsampled])
    
    # Shuffle the dataset to mix classes
    # Important: Q-Learning is sensitive to order of examples
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"  Balanced dataset: {len(df_balanced)} transactions (50/50 split)")
    
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