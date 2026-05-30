# src/preprocess.py
"""
Pipeline preprocessing data time series untuk prediksi harga kebutuhan pokok.
  1. Merge semua sumber data
  2. Handle missing values
  3. Feature engineering
  4. Normalisasi
  5. Buat sliding window sequences untuk model
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config import SEQ_LEN, PRED_LEN, TRAIN_RATIO, DATA_PROCESSED_DIR


# ── 1. Merge semua sumber ─────────────────────────────────────────────────

def merge_all_sources(data: dict) -> pd.DataFrame:
    """Gabungkan semua DataFrame ke satu tabel harian."""
    print("[INFO] Menggabungkan semua sumber data...")

    local    = data["local"]
    external = data["external"]
    weather  = data["weather"]
    holidays = data["holidays"]

    df = local.copy()
    df = df.join(external,  how="left")
    df = df.join(weather,   how="left")
    df = df.join(holidays,  how="left")

    print(f"  [OK] Shape setelah merge: {df.shape}")
    return df


# ── 2. Handle missing values ──────────────────────────────────────────────

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill (akhir pekan/libur), lalu backward-fill sisa awal."""
    print(f"[INFO] Missing values sebelum: {df.isnull().sum().sum()}")
    df = df.ffill().bfill()
    print(f"  [OK] Missing values sesudah: {df.isnull().sum().sum()}")
    return df


# ── 3. Feature engineering ────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan fitur teknikal dan temporal."""
    print("[INFO] Feature engineering...")

    from config import COMMODITIES
    target_cols = COMMODITIES

    for col in target_cols:
        if col not in df.columns:
            continue
        # Lag features
        for lag in [1, 7, 14, 30]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
        # Rolling statistics
        df[f"{col}_ma7"]  = df[col].rolling(7).mean()
        df[f"{col}_ma30"] = df[col].rolling(30).mean()
        df[f"{col}_std7"] = df[col].rolling(7).std()
        # Rate of change
        df[f"{col}_roc7"] = df[col].pct_change(7)

    # External macroeconomic and weather lag + rolling features (delayed price transmission)
    external_cols = ["USD_IDR", "Minyak", "Gandum", "Kedelai", "Jagung", "curah_hujan_mm", "suhu_c"]
    for col in external_cols:
        if col not in df.columns:
            continue
        # Delayed transmission effect lags
        for lag in [7, 14, 30]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
        # Smoothing trends
        df[f"{col}_ma7"]  = df[col].rolling(7).mean()
        df[f"{col}_ma30"] = df[col].rolling(30).mean()

    # Temporal features
    df["sin_month"] = np.sin(2 * np.pi * df.index.month / 12)
    df["cos_month"] = np.cos(2 * np.pi * df.index.month / 12)
    df["sin_dow"]   = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["cos_dow"]   = np.cos(2 * np.pi * df.index.dayofweek / 7)

    df = df.dropna()
    print(f"  [OK] Shape setelah feature engineering: {df.shape}")
    return df


# ── 4. Normalisasi ────────────────────────────────────────────────────────

def normalize(df: pd.DataFrame):
    """MinMax scale ke [0,1]. Kembalikan array + scaler untuk inverse."""
    print("[INFO] Normalisasi data...")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)

    # Simpan scaler & kolom untuk inverse transform nanti
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    with open(os.path.join(DATA_PROCESSED_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump({"scaler": scaler, "columns": df.columns.tolist()}, f)
    with open(os.path.join(DATA_PROCESSED_DIR, "feature_columns.txt"), "w") as f:
        f.write("\n".join(df.columns.tolist()))

    print(f"  [OK] Scaler tersimpan di {DATA_PROCESSED_DIR}/scaler.pkl")
    return scaled, scaler, df.columns.tolist()


# ── 5. Sliding window sequences ───────────────────────────────────────────

def create_sequences(scaled: np.ndarray, target_idx: int = 0):
    """
    Buat pasangan (X, y) dengan sliding window.
    X : (N, SEQ_LEN, n_features)
    y : (N, PRED_LEN)  — target kolom pertama (Beras by default)
    """
    print(f"[INFO] Membuat sequences (seq={SEQ_LEN}, pred={PRED_LEN})...")
    X, y = [], []
    total = len(scaled)
    for i in range(total - SEQ_LEN - PRED_LEN + 1):
        X.append(scaled[i : i + SEQ_LEN])
        y.append(scaled[i + SEQ_LEN : i + SEQ_LEN + PRED_LEN, target_idx])
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    print(f"  [OK] X: {X.shape} | y: {y.shape}")
    return X, y


# ── 6. Train / validation / test split ───────────────────────────────────

def split_data(X: np.ndarray, y: np.ndarray):
    n       = len(X)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * ((1 - TRAIN_RATIO) / 2))

    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val   = X[n_train : n_train + n_val]
    y_val   = y[n_train : n_train + n_val]
    X_test  = X[n_train + n_val:]
    y_test  = y[n_train + n_val:]

    print(f"  [OK] Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Simpan ke disk
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    np.save(os.path.join(DATA_PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(DATA_PROCESSED_DIR, "X_val.npy"),   X_val)
    np.save(os.path.join(DATA_PROCESSED_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(DATA_PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(DATA_PROCESSED_DIR, "y_val.npy"),   y_val)
    np.save(os.path.join(DATA_PROCESSED_DIR, "y_test.npy"),  y_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Pipeline utama ────────────────────────────────────────────────────────

def run_preprocessing(data: dict, target_commodity: str = "Beras"):
    df       = merge_all_sources(data)
    df       = handle_missing(df)
    df       = add_features(df)
    scaled, scaler, cols = normalize(df)

    target_idx = cols.index(target_commodity) if target_commodity in cols else 0
    X, y       = create_sequences(scaled, target_idx=target_idx)
    splits     = split_data(X, y)

    # Simpan y untuk semua komoditas
    from config import COMMODITIES
    n       = len(X)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * ((1 - TRAIN_RATIO) / 2))
    
    for com in COMMODITIES:
        t_idx = cols.index(com) if com in cols else 0
        _, y_com = create_sequences(scaled, target_idx=t_idx)
        np.save(os.path.join(DATA_PROCESSED_DIR, f"y_train_{com}.npy"), y_com[:n_train])
        np.save(os.path.join(DATA_PROCESSED_DIR, f"y_val_{com}.npy"),   y_com[n_train : n_train + n_val])
        np.save(os.path.join(DATA_PROCESSED_DIR, f"y_test_{com}.npy"),  y_com[n_train + n_val:])

    print(f"\n[SUCCESS] Preprocessing selesai. Target default: {target_commodity}. Seluruh komoditas disimpan.")
    return splits, scaler, cols


if __name__ == "__main__":
    from scraper import collect_all_data
    data = collect_all_data()
    run_preprocessing(data)

