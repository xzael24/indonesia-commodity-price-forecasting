# config.py — Semua konfigurasi project ada di sini

import os
import pandas as pd

# ── Path ──────────────────────────────────────────────────────────────────
DATA_RAW_DIR       = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
MODEL_DIR          = "models"

# ── Komoditas yang diprediksi ──────────────────────────────────────────────
_fallback_commodities = ["Beras", "Minyak Goreng", "Telur Ayam", "Cabai Merah", "Daging Ayam"]
_pihps_path = os.path.join(DATA_RAW_DIR, "pihps_harga.csv")

try:
    if os.path.exists(_pihps_path):
        _df_cols = pd.read_csv(_pihps_path, nrows=0).columns.tolist()
        COMMODITIES = [c for c in _df_cols if c and c.strip() and c.lower() not in ['date', 'unnamed: 0', 'index', 'tanggal']]
        if not COMMODITIES:
            COMMODITIES = _fallback_commodities
    else:
        COMMODITIES = _fallback_commodities
except Exception:
    COMMODITIES = _fallback_commodities


# ── Ticker Yahoo Finance untuk faktor eksternal ───────────────────────────
YAHOO_TICKERS = {
    "Gandum"  : "ZW=F",
    "Kedelai" : "ZS=F",
    "Jagung"  : "ZC=F",
    "Minyak"  : "CL=F",   # Crude Oil
    "USD_IDR" : "IDR=X",
}

# ── Tanggal data ──────────────────────────────────────────────────────────
START_DATE = "2020-01-01"

# ── Model hyperparameter ──────────────────────────────────────────────────
SEQ_LEN     = 30    # panjang lookback window (hari)
PRED_LEN    = 7     # horizon prediksi (hari)
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.2
EPOCHS      = 100
LR          = 0.001
BATCH_SIZE  = 32
TRAIN_RATIO = 0.8

