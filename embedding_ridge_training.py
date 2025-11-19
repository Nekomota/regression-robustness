
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer
import time

# Load Data
df = pd.read_parquet("preprocessed/royalroad_cleaned_Version2.parquet")
print("Data Loaded.")

train_df, val_df = train_test_split(df, test_size=0.2, random_state=1)

# Numeric Features
num_cols = [
"title_char_len", "title_token_len",
"syn_char_len", "syn_token_len",
"title_exclaim", "title_question", "title_ellipses",
"syn_exclaim", "syn_question", "syn_ellipses", "syn_newlines",
"syn_unique_tokens", "syn_ttr", "syn_avg_token_len"
]

X_num_train = train_df[num_cols].fillna(0).to_numpy(dtype=np.float64)
X_num_val   = val_df[num_cols].fillna(0).to_numpy(dtype=np.float64)

scaler = StandardScaler()
X_num_train = scaler.fit_transform(X_num_train)
X_num_val   = scaler.transform(X_num_val)


# Sentence Embeddings
print("Loading Sentence-BERT model...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')  # ~384-dim vectors

print("Encoding Titles...")
X_title_train = embed_model.encode(train_df['Title'].fillna("").tolist(), batch_size=64, show_progress_bar=True)
X_title_val   = embed_model.encode(val_df['Title'].fillna("").tolist(), batch_size=64, show_progress_bar=True)

print("Encoding Synopses...")
X_syn_train = embed_model.encode(train_df['Synopsis'].fillna("").tolist(), batch_size=64, show_progress_bar=True)
X_syn_val   = embed_model.encode(val_df['Synopsis'].fillna("").tolist(), batch_size=64, show_progress_bar=True)


# Concatenate Features
X_train = np.hstack([X_title_train, X_syn_train, X_num_train])
X_val   = np.hstack([X_title_val, X_syn_val, X_num_val])

print("Final feature shapes: X_train:", X_train.shape, "X_val:", X_val.shape)

# Log-transform Target
y_train_raw = train_df['Followers'].to_numpy(dtype=np.float64)
y_val_raw   = val_df['Followers'].to_numpy(dtype=np.float64)

y_train_log = np.log1p(y_train_raw)
y_val_log   = np.log1p(y_val_raw)


# Ridge Regression
ridge_model = Ridge(alpha=10)

print("\n===== Ridge Regression (Embeddings + Numeric Features) =====")
start = time.time()
ridge_model.fit(X_train, y_train_log)
end = time.time()
print(f"Training time: {end - start:.3f} sec")

# Predictions
pred_log = ridge_model.predict(X_val)
pred_raw = np.expm1(pred_log)
pred_raw = np.clip(pred_raw, 0, None)

# Evaluation 

mae = mean_absolute_error(y_val_raw, pred_raw)
rmse = np.sqrt(mean_squared_error(y_val_raw, pred_raw))
r2 = r2_score(y_val_raw, pred_raw)

print(f"MAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²  : {r2:.4f}")


# bucket Evaluation
buckets = [0, 20, 200, 1000, np.inf]  # <20, <200, <1000, >1000
y_val_bucket = np.digitize(y_val_raw, bins=buckets) - 1
pred_bucket  = np.digitize(pred_raw, bins=buckets) - 1

acc = accuracy_score(y_val_bucket, pred_bucket)
bal_acc = balanced_accuracy_score(y_val_bucket, pred_bucket)
print(f"\nBucketed Accuracy: {acc*100:.2f}%")
print(f"Bucketed Balanced Accuracy: {bal_acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_val_bucket, pred_bucket))
print("\nConfusion Matrix:")
print(confusion_matrix(y_val_bucket, pred_bucket))
