#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
News Classification System

This script implements a comprehensive news classification system that can:
1. Load and preprocess news data from multiple sources
2. Perform data analysis and visualization
3. Apply different sampling techniques
4. Extract features using TF-IDF or Sentence Transformers
5. Train and evaluate multiple classification models

The system supports three types of news: true, fake, and AI-generated.
"""

import os, random, numpy as np, torch
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)   # cuDNN deterministic

import argparse
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sentence_transformers import SentenceTransformer, models
from datasets import load_dataset

def clean_text(text):
    """
    Clean text data by removing special characters and extra spaces.
    
    Args:
        text (str): Input text to be cleaned
        
    Returns:
        str: Cleaned text
    """
    if text is None:
        return ""
    text = text.replace('\xa0', ' ').replace('&nbsp;', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def delete_prefix(text):
    """
    Remove prefix from text if it exists.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text without prefix
    """
    if pd.isna(text):
        return ""
    if '-' in text:
        text = text.split('-', 1)[1]
    return text

def load_and_preprocess_data():
    """
    Load and preprocess data from multiple sources.
    
    Returns:
        tuple: Three DataFrames containing true, fake, and AI-generated news
    """
    # Load local data
    df_fake = pd.read_csv('fake.csv')
    df_true = pd.read_csv('true.csv')
    
    # Remove unnecessary columns
    df_fake.drop(labels=['title', 'subject', 'date'], axis=1, inplace=True)
    df_true.drop(labels=['title', 'subject', 'date'], axis=1, inplace=True)
    
    # Clean text
    df_true['text'] = df_true['text'].apply(delete_prefix)
    df_true['text'] = df_true['text'].apply(clean_text)
    df_fake['text'] = df_fake['text'].apply(clean_text)
    
    # Load additional dataset
    ds = load_dataset("tum-nlp/neural-news-benchmark")
    LABEL_MAP = {"neural": 1, "real": 0}
    
    def process_split(split_name):
        return [
            {
                "label": LABEL_MAP[item["label"]],
                "text": clean_text(item["body"])
            }
            for item in ds[split_name] 
            if item["body"] is not None and item["language"] == "en"
        ]
    
    # Process all data
    all_data = process_split('train') + process_split('validation') + process_split('test')
    df = pd.DataFrame(all_data)
    
    # Separate data
    df_neural = df[df['label'] == 1].reset_index(drop=True)
    df_real = df[df['label'] == 0].reset_index(drop=True)

    df_neural = df_neural.drop(columns=['label'])
    df_real   = df_real.drop(columns=['label'])
    
    # Merge data
    df_true = pd.concat([df_true, df_real], ignore_index=True)
    
    # Remove duplicates
    df_true = df_true.drop_duplicates(subset='text').reset_index(drop=True)
    df_fake = df_fake.drop_duplicates(subset='text').reset_index(drop=True)
    df_neural = df_neural.drop_duplicates(subset='text').reset_index(drop=True)
    
    # Add source labels
    df_true['source'] = 'true'
    df_fake['source'] = 'fake'
    df_neural['source'] = 'neural'
    
    return df_true, df_fake, df_neural

def sample_data(df_true, df_fake, df_neural, sampling_method):
    """
    Apply sampling method to balance the dataset.
    
    Args:
        df_true (DataFrame): True news data
        df_fake (DataFrame): Fake news data
        df_neural (DataFrame): AI-generated news data
        sampling_method (str): Sampling method to use ('none', 'undersample', 'oversample')
        
    Returns:
        DataFrame: Balanced dataset
    """
    # Reset indices to ensure consistent RNG input
    df_true   = df_true.reset_index(drop=True)
    df_fake   = df_fake.reset_index(drop=True)
    df_neural = df_neural.reset_index(drop=True)

    if sampling_method == 'undersample':
        k = min(len(df_true), len(df_fake), len(df_neural))
        df_true   = df_true.sample(n=k, random_state=SEED)
        df_fake   = df_fake.sample(n=k, random_state=SEED)
        df_neural = df_neural.sample(n=k, random_state=SEED)

    elif sampling_method == 'oversample':
        k = max(len(df_true), len(df_fake), len(df_neural))
        df_true   = df_true.sample(n=k, replace=True, random_state=SEED)
        df_fake   = df_fake.sample(n=k, replace=True, random_state=SEED)
        df_neural = df_neural.sample(n=k, replace=True, random_state=SEED)

    # Concatenate and shuffle if sampling was applied
    df_out = pd.concat([df_true, df_fake, df_neural], ignore_index=True)
    if sampling_method != 'none':
        df_out = df_out.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df_out

def get_features(df, feature_method):
    """
    Extract features from text data using specified method.
    
    Args:
        df (DataFrame): Input data
        feature_method (str): Feature extraction method ('tfidf' or 'sentence_transformer')
        
    Returns:
        array: Feature matrix
    """
    if feature_method == 'tfidf':
        vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
        X_tfidf = vectorizer.fit_transform(df['text'])
        return X_tfidf
    elif feature_method == 'sentence_transformer':
        # Use pre-trained model (CPU-friendly)
        model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        # Get embeddings (512-dimensional, mean pooling)
        return model.encode(df['text'].tolist(), show_progress_bar=True)
    else:
        raise ValueError(f"Unknown feature method: {feature_method}")

def train_and_evaluate(X_train, X_test, y_train, y_test, model_type):
    """
    Train and evaluate a classification model.
    
    Args:
        X_train (array): Training features
        X_test (array): Test features
        y_train (array): Training labels
        y_test (array): Test labels
        model_type (str): Type of model to train
        
    Returns:
        tuple: Trained model and predictions
    """
    if model_type == 'logistic_regression':
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == 'svm':
        model = LinearSVC(random_state=42, dual=True)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{model_type.upper()} Model Results:")
    print("=" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model, y_pred

def analyze_data(df_true, df_fake, df_neural):
    """
    Analyze data and generate visualizations.
    
    Args:
        df_true (DataFrame): True news data
        df_fake (DataFrame): Fake news data
        df_neural (DataFrame): AI-generated news data
        
    Returns:
        tuple: Filtered DataFrames
    """
    # Calculate text lengths
    df_true['length'] = df_true['text'].str.len()
    df_fake['length'] = df_fake['text'].str.len()
    df_neural['length'] = df_neural['text'].str.len()
    
    # Print length statistics
    def print_stats(name, df):
        print(f"\n{name} Text Length Statistics:")
        print(df['length'].describe())
    
    print_stats('True News', df_true)
    print_stats('Fake News', df_fake)
    print_stats('AI-Generated News', df_neural)
    
    # Filter text length
    min_length = 750
    max_length = 5000
    
    df_true = df_true[df_true['text'].str.len().between(min_length, max_length)].reset_index(drop=True)
    df_fake = df_fake[df_fake['text'].str.len().between(min_length, max_length)].reset_index(drop=True)
    df_neural = df_neural[df_neural['text'].str.len().between(min_length, max_length)].reset_index(drop=True)
    
    # Print filtered data information
    print("\nFiltered Data Information:")
    print(f"True News Count: {len(df_true)}")
    print(f"Fake News Count: {len(df_fake)}")
    print(f"AI-Generated News Count: {len(df_neural)}")
    
    # Plot text length distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df_true['length'], bins=50, alpha=0.5, label='True News')
    plt.hist(df_fake['length'], bins=50, alpha=0.5, label='Fake News')
    plt.hist(df_neural['length'], bins=50, alpha=0.5, label='AI-Generated News')
    plt.legend()
    plt.title("Text Length Distribution")
    plt.xlabel("Text Length")
    plt.ylabel("Frequency")
    plt.show()
    
    return df_true, df_fake, df_neural

def visualize_data(X, y, le):
    """
    Visualize data distribution using PCA and KMeans.
    
    Args:
        X (array): Feature matrix
        y (array): Labels
        le (LabelEncoder): Label encoder
    """
    # Apply PCA for dimensionality reduction
    X_pca = PCA(n_components=2).fit_transform(X.toarray() if hasattr(X, 'toarray') else X)
    
    # Create DataFrame for visualization
    df_vis = pd.DataFrame({
        'x': X_pca[:, 0],
        'y': X_pca[:, 1],
        'source': le.inverse_transform(y)
    })
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    df_vis['cluster'] = kmeans.fit_predict(X)
    
    # Plot true label distribution
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_vis, x='x', y='y', hue='source', 
                   palette='Set1', s=50)
    plt.title("True Label Distribution (PCA 2D)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(title='Source')
    plt.grid(True)
    plt.show()
    
    # Plot KMeans clustering results
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_vis, x='x', y='y', hue='cluster', 
                   palette='Set2', s=50)
    plt.title("KMeans Clustering Results (PCA 2D)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.show()

def main():
    """
    Main function to run the news classification system.
    """
    parser = argparse.ArgumentParser(description='News Classification')
    parser.add_argument('--sampling', choices=['none', 'undersample', 'oversample'], 
                      default='none', help='Sampling method to use')
    parser.add_argument('--features', choices=['tfidf', 'sentence_transformer'], 
                      default='tfidf', help='Feature extraction method')
    parser.add_argument('--model', choices=['logistic_regression', 'svm', 'random_forest'], 
                      default='logistic_regression', help='Model to use')
    
    args = parser.parse_args()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df_true, df_fake, df_neural = load_and_preprocess_data()
    
    # Print initial data information
    print("\nInitial Data Information:")
    print("True News Data:")
    print(df_true.info())
    print("\nFake News Data:")
    print(df_fake.info())
    print("\nAI-Generated News Data:")
    print(df_neural.info())
    
    # Analyze data and generate visualizations
    print("\nAnalyzing data...")
    df_true, df_fake, df_neural = analyze_data(df_true, df_fake, df_neural)
    
    # Print analyzed data information
    print("\nData Information After Analysis:")
    print("True News Data:")
    print(df_true.info())
    print("\nFake News Data:")
    print(df_fake.info())
    print("\nAI-Generated News Data:")
    print(df_neural.info())
    
    # Sample data
    print(f"\nUsing {args.sampling} sampling method...")
    df_balanced = sample_data(df_true, df_fake, df_neural, args.sampling)
    
    # Prepare labels
    le = LabelEncoder()
    y = le.fit_transform(df_balanced['source'])
    
    # Extract features
    print(f"\nUsing {args.features} for feature extraction...")
    X = get_features(df_balanced, args.features)
    
    # Visualize data distribution
    print("\nVisualizing data distribution...")
    visualize_data(X, y, le)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train and evaluate model
    print(f"\nTraining {args.model} model...")
    model, y_pred = train_and_evaluate(X_train, X_test, y_train, y_test, args.model)

if __name__ == "__main__":
    main() 