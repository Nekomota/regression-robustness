#imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

# options
trainOLS = False
numTitleFeatures = 5000
numSynopsisFeatures = 20000


# helper to evaluate a model

def evaluate_model(name, model, X_train, y_train_log, X_val, y_val_log, y_val_raw):
    print(f"\n===== {name} =====")
    
    start = time.time()
    model.fit(X_train, y_train_log)
    end = time.time()
    
    print(f"Training time: {end - start:.3f} seconds")

    pred_log = model.predict(X_val)
    pred_raw = np.expm1(pred_log)
    pred_raw = np.clip(pred_raw, 0, None)

    # calculate a few different metrics
    mae = mean_absolute_error(y_val_raw, pred_raw)
    rmse = np.sqrt(mean_squared_error(y_val_raw, pred_raw))
    r2 = r2_score(y_val_raw, pred_raw)

    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")
    return pred_raw

# Load preprocessed data
df = pd.read_parquet("preprocessed/royalroad_cleaned_Version2.parquet")
print("Data Loaded.")

# Split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=1)

# TF-IDF vectorizers
title_vec = TfidfVectorizer(max_features=numTitleFeatures, ngram_range=(1,2))
syn_vec = TfidfVectorizer(max_features=numSynopsisFeatures, ngram_range=(1,2))

# fit/transform title TF-IDF on training titles
print("Fitting on titles...")
X_title_train = title_vec.fit_transform(train_df["Title"].fillna("").astype(str))
X_title_val   = title_vec.transform(val_df["Title"].fillna("").astype(str))

# fit/transform synopsis TF-IDF on training synopses
print("Fitting on synopsises...")
X_syn_train = syn_vec.fit_transform(train_df["Synopsis"].fillna("").astype(str))
X_syn_val   = syn_vec.transform(val_df["Synopsis"].fillna("").astype(str))

print("TF-IDF shapes: title_train:", X_title_train.shape, "syn_train:", X_syn_train.shape)

# numeric features: 
from sklearn.preprocessing import StandardScaler

num_cols = [
    "title_char_len", "title_token_len",
    "syn_char_len", "syn_token_len",
    "title_exclaim", "title_question", "title_ellipses",
    "syn_exclaim", "syn_question", "syn_ellipses", "syn_newlines",
    "syn_unique_tokens", "syn_ttr", "syn_avg_token_len"
]

# extract numeric arrays 
X_num_train_raw = train_df[num_cols].fillna(0).to_numpy(dtype=np.float64)
X_num_val_raw   = val_df[num_cols].fillna(0).to_numpy(dtype=np.float64)

# scale numeric features 
scaler = StandardScaler()
X_num_train = scaler.fit_transform(X_num_train_raw)
X_num_val   = scaler.transform(X_num_val_raw)

print("Numeric shapes: train:", X_num_train.shape, "val:", X_num_val.shape)

# convert numeric features to sparse matrix
from scipy.sparse import csr_matrix, hstack

# convert numeric arrays to sparse CSR matrices
X_num_train_sp = csr_matrix(X_num_train)
X_num_val_sp   = csr_matrix(X_num_val)

# horizontally stack: [title_tfidf | syn_tfidf | numeric_features]
X_train = hstack([X_title_train, X_syn_train, X_num_train_sp], format="csr")
X_val   = hstack([X_title_val,   X_syn_val,   X_num_val_sp],   format="csr")

print("Final concatenated feature shapes: X_train:", X_train.shape, "X_val:", X_val.shape)

# target vectors

import numpy as np

# raw target
y_train_raw = train_df["Followers"].to_numpy(dtype=np.float64)
y_val_raw   = val_df["Followers"].to_numpy(dtype=np.float64)

# log-transformed target (recommended for skewed counts)
y_train_log = np.log1p(y_train_raw)
y_val_log   = np.log1p(y_val_raw)

print("Targets prepared: sample counts:", len(y_train_raw), len(y_val_raw))

#test on no reg, l1, l2
if trainOLS:
    ols_model = LinearRegression()
    pred_ols = evaluate_model(
        "Linear Regression (OLS)",
        ols_model,
        X_train, y_train_log,
        X_val, y_val_log,
        y_val_raw
    )

lasso_model = Lasso(alpha=10)
pred_lasso = evaluate_model(
    "Lasso Regression (L1)",
    lasso_model,
    X_train, y_train_log,
    X_val, y_val_log,
    y_val_raw
)

ridge_model = Ridge(alpha=10)
pred_ridge = evaluate_model(
    "Ridge Regression (L2)",
    ridge_model,
    X_train, y_train_log,
    X_val, y_val_log,
    y_val_raw
)

from sklearn.linear_model import ElasticNet

elastic_model = ElasticNet(alpha=10, l1_ratio=0.2)
pred_elastic = evaluate_model(
    "ElasticNet Regression",
    elastic_model,
    X_train, y_train_log,
    X_val, y_val_log,
    y_val_raw
)



