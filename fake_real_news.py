# -*- coding: utf-8 -*-
"""
This script processes and classifies news articles as real, fake, or neural-generated.
It performs the following major tasks:
1. Loads and cleans datasets from local CSV files and the HuggingFace dataset hub.
2. Preprocesses text: removes prefixes, whitespace normalization, and HTML replacements.
3. Performs exploratory analysis and visualization of text length distributions.
4. Vectorizes the text using TF-IDF and SentenceTransformer embeddings.
5. Applies clustering (KMeans) and dimensionality reduction (PCA) for visualization.
6. Trains and evaluates multiple classifiers: Logistic Regression, SVM, and Random Forest.
7. Applies resampling methods for class balance (undersampling, oversampling).

Generated and revised from Colab: https://colab.research.google.com/drive/1MMZnpWYKmyc44Ow-RwJdlc6hpHtY-MO3
"""

import pandas as pd

# Load fake and true news datasets from the current directory
df_fake = pd.read_csv('fake.csv')
df_true = pd.read_csv('true.csv')

# Print a preview to confirm successful loading
print("✅ Fake.csv loaded:")
print(df_fake.head())

print("\n✅ True.csv loaded:")
print(df_true.head())

# Drop irrelevant metadata columns
for df in [df_fake, df_true]:
    df.drop(labels=['title', 'subject', 'date'], axis=1, inplace=True)


import re


# Function to clean unwanted characters and whitespace
def clean_text(text):
    if text is None:
        return ""
    text = text.replace('\xa0', ' ').replace('&nbsp;', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Function to delete prefixes before the first hyphen
def delete_prefix(text):
    if pd.isna(text):
        return ""
    if '-' in text:
        text = text.split('-', 1)[1]
    return text

# Apply text preprocessing functions to both true and fake datasets
df_true['text'] = df_true['text'].apply(delete_prefix).apply(clean_text)
df_fake['text'] = df_fake['text'].apply(clean_text)

# Print dataset info after preprocessing
print("\n✅ df_true.info():")
print(df_true.info())
print("\n✅ df_fake.info():")
print(df_fake.info())

# Load additional dataset from HuggingFace: Neural News Benchmark
try:
    from datasets import load_dataset
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'datasets'])
    from datasets import load_dataset

import re

# Load the Neural News Benchmark dataset
ds = load_dataset("tum-nlp/neural-news-benchmark")

# Print a preview of the dataset
print(ds["train"])

# Define the label mapping
LABEL_MAP = {
    "neural": 1,
    "real": 0
}


# Function to filter and process the dataset
def process_split(split_name):
    processed_data = [
        {
            "label": LABEL_MAP[item["label"]],
            "text": clean_text(item["body"])
        }
        for item in ds[split_name] if item["body"] is not None and item["language"] == "en"
    ]
    return processed_data

# Example usage
test_data = process_split('test')
train_data = process_split('train')
valid_data = process_split('validation')
all_data = train_data + valid_data + test_data
df = pd.DataFrame(all_data)


# Group by label and print first few rows
grouped = df.groupby('label')

# Split into neural (label=1) and real (label=0)
df_neural = df[df['label'] == 1].reset_index(drop=True)
df_real = df[df['label'] == 0].reset_index(drop=True)


# Drop label column
df_neural = df_neural.drop(columns=['label'])
df_real = df_real.drop(columns=['label'])

# Merge df_true and df_real
df_true = pd.concat([df_true, df_real], ignore_index=True)

# Drop duplicates
df_true = df_true.drop_duplicates(subset='text').reset_index(drop=True)
df_fake = df_fake.drop_duplicates(subset='text').reset_index(drop=True)
df_neural = df_neural.drop_duplicates(subset='text').reset_index(drop=True)


df_true['length'] = df_true['text'].str.len()
df_fake['length'] = df_fake['text'].str.len()
df_neural['length'] = df_neural['text'].str.len()

def print_stats(name, df):
    print(f"{name} Length Stats:")
    print(df['length'].describe())
    print()

print_stats('True', df_true)
print_stats('Fake', df_fake)
print_stats('Neural', df_neural)

# Filter out entries with text length greater than 5000

df_true = df_true[df_true['text'].str.len() <= 5000].reset_index(drop=True)
df_fake = df_fake[df_fake['text'].str.len() <= 5000].reset_index(drop=True)
df_neural = df_neural[df_neural['text'].str.len() <= 5000].reset_index(drop=True)

# Filter out entries with text length less than 750
df_true = df_true[df_true['text'].str.len() >= 750].reset_index(drop=True)
df_fake = df_fake[df_fake['text'].str.len() >= 750].reset_index(drop=True)
df_neural = df_neural[df_neural['text'].str.len() >= 750].reset_index(drop=True)

print(df_true.info())
print(df_fake.info())
print(df_neural.info())


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(df_true['length'], bins=50, alpha=0.5, label='True')
plt.hist(df_fake['length'], bins=50, alpha=0.5, label='Fake')
plt.hist(df_neural['length'], bins=50, alpha=0.5, label='Neural')
plt.legend()
plt.title("Text Length Distribution")
plt.xlabel("Text Length")
plt.ylabel("Frequency")
plt.show()

# Add source labels
df_true['source'] = 'true'
df_fake['source'] = 'fake'
df_neural['source'] = 'neural'

# Merge all data
df_all = pd.concat([df_true, df_fake, df_neural], ignore_index=True)

# Ensure text column is a string
df_all['text'] = df_all['text'].astype(str)

df_all.head()


from sklearn.feature_extraction.text import TfidfVectorizer

# Use default English stop word list
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_tfidf = vectorizer.fit_transform(df_all['text'])

from sklearn.cluster import KMeans

num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
df_all['cluster'] = kmeans.fit_predict(X_tfidf)

print(df_all.groupby(['cluster', 'source']).size())


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Perform PCA dimensionality reduction to 2D for visualization
X_pca = PCA(n_components=2).fit_transform(X_tfidf.toarray())

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Append PCA components as new columns in the dataframe for plotting
df_all['x'] = X_pca[:, 0]
df_all['y'] = X_pca[:, 1]

# Visualize KMeans clustering results in 2D PCA space
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_all, x='x', y='y', hue='cluster', palette='Set2', s=50)
plt.title("KMeans Results")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title='Cluster')
plt.show()

# Visualize true labels (source) in 2D PCA space
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_all, x='x', y='y', hue='source', palette='Set1', s=50)
plt.title("Real Results")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title='Source')
plt.show()

from sklearn.preprocessing import LabelEncoder

# Convert string labels (true/fake/neural) into numerical format: 0/1/2
le = LabelEncoder()
y = le.fit_transform(df_all['source'])  # y is the target label

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_pred = log_reg.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_, digits=3))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

from sklearn.svm import LinearSVC

svm_clf = LinearSVC(random_state=42)
svm_clf.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_pred_svm = svm_clf.predict(X_test)

print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("\nSVM Classification Report:\n", classification_report(y_test, y_pred_svm, target_names=le.classes_, digits=3))
print("\nSVM Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))

from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_clf.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_pred_rf = rf_clf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf, target_names=le.classes_, digits=3))
print("\nRandom Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

#..................................
from sentence_transformers import SentenceTransformer

# Use pre-trained model (suitable for CPU)
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")


# Extract text and labels from the previously downsampled df_balanced
texts = df_all['text'].tolist()
labels = le.fit_transform(df_all['source'])  # source → 0/1/2

# Get embeddings (512-dimensional, average pooling)
embeddings = model.encode(texts, show_progress_bar=True)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.2, random_state=42, stratify=labels
)

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_pred = clf.predict(X_test)

print("Embedding Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_, digits=3))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

from sklearn.svm import LinearSVC, SVC

svm_clf = LinearSVC(random_state=42)
svm_clf.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_pred_svm = svm_clf.predict(X_test)

print("Embedding SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("\nSVM Classification Report:\n", classification_report(y_test, y_pred_svm, target_names=le.classes_, digits=3))
print("\nSVM Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Train model
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_clf.fit(X_train, y_train)

# Predict and evaluate
y_pred_rf = rf_clf.predict(X_test)

print("Embedding Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report (RF):\n", classification_report(y_test, y_pred_rf, target_names=le.classes_, digits=3))
print("\nConfusion Matrix (RF):\n", confusion_matrix(y_test, y_pred_rf))
#.............................

# Downsample

df_true_sampled = df_true.sample(n=3125, random_state=42)
df_fake_sampled = df_fake.sample(n=3125, random_state=42)
df_neural_sampled = df_neural.copy()  # 已经是最小类


# Upsample
'''
df_neural_sampled = df_neural.sample(n=17313, replace=True, random_state=42)
df_fake_sampled = df_fake.sample(n=17313, replace=True, random_state=42)
df_true_sampled = df_true.copy()
'''
# No sampling
'''
df_neural_sampled = df_neural.copy()
df_fake_sampled = df_fake.copy()
df_true_sampled = df_true.copy()
'''

df_balanced = pd.concat([df_true_sampled, df_fake_sampled, df_neural_sampled], ignore_index=True).sample(frac=1, random_state=42)

print(df_balanced['source'].value_counts())

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_tfidf = vectorizer.fit_transform(df_balanced['text'])

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(df_balanced['source'])  # source → 0/1/2

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_pred = log_reg.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_, digits=3))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

from sklearn.svm import LinearSVC

svm_clf = LinearSVC(random_state=42)
svm_clf.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_pred_svm = svm_clf.predict(X_test)

print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("\nSVM Classification Report:\n", classification_report(y_test, y_pred_svm, target_names=le.classes_, digits=3))
print("\nSVM Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# PCA dimensionality reduction for visualization
X_pca = PCA(n_components=2).fit_transform(X_test.toarray())

df_vis_svm = pd.DataFrame({
    'pca_x': X_pca[:, 0],
    'pca_y': X_pca[:, 1],
    'true_label': le.inverse_transform(y_test),
    'pred_label': le.inverse_transform(y_pred_svm)
})

# True labels
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df_vis_svm, x='pca_x', y='pca_y', hue='true_label', palette='Set1', alpha=0.7)
plt.title("SVM - True Labels (PCA 2D)")
plt.grid(True)
plt.show()

# SVM Predicted Labels
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df_vis_svm, x='pca_x', y='pca_y', hue='pred_label', palette='Set2', alpha=0.7)
plt.title("SVM - Predicted Labels (PCA 2D)")
plt.grid(True)
plt.show()

from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_clf.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_pred_rf = rf_clf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf, target_names=le.classes_, digits=3))
print("\nRandom Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
