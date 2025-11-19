import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

from scipy.sparse import csr_matrix, hstack

# Bucket Function

def bucketize(x):
    if x < 20:
        return 0         # small
    elif x < 200:
        return 1         # medium
    elif x < 1000:
        return 2         # big
    else:
        return 3         # extremely big


# Config

numTitleFeatures = 5000
numSynopsisFeatures = 20000

# Load Preprocessed Data
df = pd.read_parquet("preprocessed/royalroad_cleaned_Version2.parquet")
print("Data Loaded.")

train_df, val_df = train_test_split(df, test_size=0.2, random_state=1)

# TF-IDF Vectorizers
title_vec = TfidfVectorizer(max_features=numTitleFeatures, ngram_range=(1,2))
syn_vec   = TfidfVectorizer(max_features=numSynopsisFeatures, ngram_range=(1,2))

print("Fitting on titles...")
X_title_train = title_vec.fit_transform(train_df["Title"].fillna("").astype(str))
X_title_val   = title_vec.transform(val_df["Title"].fillna("").astype(str))

print("Fitting on synopses...")
X_syn_train = syn_vec.fit_transform(train_df["Synopsis"].fillna("").astype(str))
X_syn_val   = syn_vec.transform(val_df["Synopsis"].fillna("").astype(str))

print("TF-IDF shapes:", X_title_train.shape, X_syn_train.shape)


# Numeric Features
num_cols = [
    "title_char_len", "title_token_len",
    "syn_char_len", "syn_token_len",
    "title_exclaim", "title_question", "title_ellipses",
    "syn_exclaim", "syn_question", "syn_ellipses", "syn_newlines",
    "syn_unique_tokens", "syn_ttr", "syn_avg_token_len"
]

X_num_train_raw = train_df[num_cols].fillna(0).to_numpy(dtype=np.float64)
X_num_val_raw   = val_df[num_cols].fillna(0).to_numpy(dtype=np.float64)

scaler = StandardScaler()
X_num_train = scaler.fit_transform(X_num_train_raw)
X_num_val   = scaler.transform(X_num_val_raw)

X_num_train_sp = csr_matrix(X_num_train)
X_num_val_sp   = csr_matrix(X_num_val)

# Combine all feature blocks
X_train = hstack([X_title_train, X_syn_train, X_num_train_sp], format="csr")
X_val   = hstack([X_title_val,   X_syn_val,   X_num_val_sp],   format="csr")

print("Final feature matrix:", X_train.shape, X_val.shape)

# Convert Followers → Buckets
y_train = train_df["Followers"].apply(bucketize).to_numpy()
y_val   = val_df["Followers"].apply(bucketize).to_numpy()

print("\n===== Logistic Regression (Bucket Classification) =====")

clf = LogisticRegression(
    max_iter=2000,
    n_jobs=-1,
    C=1.0,
    solver="lbfgs",
)

start = time.time()
clf.fit(X_train, y_train)
end = time.time()

print(f"Training time: {end - start:.3f} sec")


pred = clf.predict(X_val)

acc = accuracy_score(y_val, pred)
bal = balanced_accuracy_score(y_val, pred)
cm = confusion_matrix(y_val, pred)

print(f"\nAccuracy: {acc*100:.2f}%")
print(f"Balanced Accuracy: {bal*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_val, pred))

print("Confusion Matrix:")
print(cm)

