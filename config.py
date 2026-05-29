# config.py — Semua konfigurasi project ada di sini

# ── Komoditas yang diprediksi ──────────────────────────────────────────────
COMMODITIES = ["Beras", "Minyak Goreng", "Telur Ayam", "Cabai Merah", "Daging Ayam"]

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

# ── Path ──────────────────────────────────────────────────────────────────
DATA_RAW_DIR       = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
MODEL_DIR          = "models"
