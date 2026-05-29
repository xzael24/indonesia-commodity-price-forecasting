# src/train.py
"""
Script utama untuk melatih dan membandingkan ketiga model.
Jalankan: python src/train.py
"""

import os
import sys
import json
import pickle
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_PROCESSED_DIR, MODEL_DIR, EPOCHS, LR, BATCH_SIZE
from src.model import get_model, train_model, evaluate_model


def load_processed_data(commodity: str = "Beras"):
    """Load data yang sudah di-preprocess dari disk untuk komoditas tertentu."""
    print(f"► Memuat data yang sudah diproses untuk {commodity}...")
    X_train = np.load(os.path.join(DATA_PROCESSED_DIR, "X_train.npy"))
    X_val   = np.load(os.path.join(DATA_PROCESSED_DIR, "X_val.npy"))
    X_test  = np.load(os.path.join(DATA_PROCESSED_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_PROCESSED_DIR, f"y_train_{commodity}.npy"))
    y_val   = np.load(os.path.join(DATA_PROCESSED_DIR, f"y_val_{commodity}.npy"))
    y_test  = np.load(os.path.join(DATA_PROCESSED_DIR, f"y_test_{commodity}.npy"))

    with open(os.path.join(DATA_PROCESSED_DIR, "scaler.pkl"), "rb") as f:
        scaler_data = pickle.load(f)

    print(f"  ✓ Input shape: {X_train.shape} | Output shape: {y_train.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler_data


def run_training():
    from config import COMMODITIES
    results = {}

    for com in COMMODITIES:
        print(f"\n{'='*50}")
        print(f"  KOMODITAS: {com.upper()}")
        print(f"{'='*50}")

        X_train, X_val, X_test, y_train, y_val, y_test, scaler_data = load_processed_data(com)
        scaler     = scaler_data["scaler"]
        cols       = scaler_data["columns"]
        input_size = X_train.shape[2]
        target_idx = cols.index(com) if com in cols else 0

        results[com] = {}

        for model_name in ["lstm", "gru", "tft"]:
            print(f"\n  Model: {model_name.upper()} ({com})")

            model   = get_model(model_name, input_size)
            history = train_model(
                model, X_train, y_train, X_val, y_val,
                epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE,
                model_name=f"{model_name}_{com}"
            )

            # Load best checkpoint
            ckpt = os.path.join(MODEL_DIR, f"{model_name}_{com}_best.pt")
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))

            print(f"\n  Evaluasi {model_name.upper()} ({com}) pada test set:")
            metrics = evaluate_model(model, X_test, y_test, scaler, target_idx=target_idx)
            results[com][model_name] = {
                "MAE" : float(metrics["MAE"]),
                "RMSE": float(metrics["RMSE"]),
                "MAPE": float(metrics["MAPE"]),
            }

    # Simpan ringkasan hasil
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*50)
    print("  RINGKASAN PERBANDINGAN MODEL")
    print("="*50)
    for com in COMMODITIES:
        print(f"\n  ► {com.upper()}")
        for name, m in results[com].items():
            print(f"    {name.upper():6s} | MAE: Rp {m['MAE']:>9,.0f} | RMSE: Rp {m['RMSE']:>9,.0f} | MAPE: {m['MAPE']:.2f}%")

    return results


if __name__ == "__main__":
    # Pastikan preprocessing sudah dijalankan
    if not os.path.exists(os.path.join(DATA_PROCESSED_DIR, "X_train.npy")):
        print("Data belum diproses. Jalankan preprocessing dulu:")
        print("  python src/preprocess.py")
        sys.exit(1)
    run_training()

