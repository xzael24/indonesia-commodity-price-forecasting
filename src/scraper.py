# src/scraper.py
"""
Modul pengambilan data dari berbagai sumber:
  - Yahoo Finance       : harga komoditas global + kurs USD/IDR
  - PIHPS Bank Indonesia: harga pangan lokal (manual export — lihat README)
  - BMKG                : data cuaca (placeholder — perlu API key BMKG)
"""

import os
import time
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime
from config import YAHOO_TICKERS, START_DATE, DATA_RAW_DIR


# ── Helper ────────────────────────────────────────────────────────────────

def _save(df: pd.DataFrame, filename: str) -> str:
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    path = os.path.join(DATA_RAW_DIR, filename)
    df.to_csv(path)
    print(f"  ✓ Tersimpan: {path} ({len(df)} baris)")
    return path


# ── 1. Yahoo Finance ──────────────────────────────────────────────────────

def fetch_yahoo_data() -> pd.DataFrame:
    """Ambil harga penutupan komoditas global & kurs dari Yahoo Finance."""
    print("► Mengambil data Yahoo Finance...")
    frames = {}
    for name, ticker in YAHOO_TICKERS.items():
        try:
            df = yf.download(ticker, start=START_DATE, progress=False, auto_adjust=True)
            if not df.empty:
                frames[name] = df["Close"].squeeze()
                print(f"  ✓ {name} ({ticker}): {len(df)} baris")
            else:
                print(f"  ✗ {name} ({ticker}): data kosong")
        except Exception as e:
            print(f"  ✗ {name} ({ticker}): error — {e}")
        time.sleep(0.5)

    combined = pd.DataFrame(frames)
    combined.index = pd.to_datetime(combined.index)
    combined = combined.resample("D").interpolate(method="time")
    _save(combined, "external_factors.csv")
    return combined


# ── 2. Data lokal PIHPS Bank Indonesia (manual export) ───────────────────

def load_bapanas_data(filepath: str = None) -> pd.DataFrame:
    """
    Load data harga lokal dari PIHPS Bank Indonesia.

    CARA MANUAL (lihat README bagian 'Setup Data PIHPS BI'):
      1. Buka https://www.bi.go.id/hargapangan
      2. Pilih menu "Tabel Harga Berdasarkan Daerah"
      3. Pilih komoditas (Beras, Minyak Goreng, Telur Ayam, Cabai Merah, Daging Ayam)
      4. Set Provinsi: "Nasional" (atau provinsi tertentu)
      5. Set Tanggal Mulai: 2020-01-01, Tanggal Selesai: hari ini
      6. Klik "Lihat Laporan" → klik tombol Export (Excel/CSV)
      7. Ulangi untuk setiap komoditas, lalu gabungkan kolomnya
      8. Rename file menjadi pihps_harga.csv
      9. Simpan di data/raw/pihps_harga.csv
      10. Jalankan ulang program ini

    Format kolom CSV yang diharapkan:
      tanggal, Beras, Minyak Goreng, Telur Ayam, Cabai Merah, Daging Ayam
    """
    # Coba load dari PIHPS dulu, fallback ke nama lama (bapanas)
    if filepath is None:
        pihps_path   = os.path.join(DATA_RAW_DIR, "pihps_harga.csv")
        bapanas_path = os.path.join(DATA_RAW_DIR, "bapanas_harga.csv")
        if os.path.exists(pihps_path):
            filepath = pihps_path
        elif os.path.exists(bapanas_path):
            filepath = bapanas_path
            print("  ℹ️  Menggunakan file lama bapanas_harga.csv. Disarankan ganti ke pihps_harga.csv.")
        else:
            filepath = pihps_path  # trigger warning di bawah

    if not os.path.exists(filepath):
        print(f"\n  ⚠️  File PIHPS belum ada di: {filepath}")
        print("  → Menggunakan data SIMULASI sebagai placeholder.")
        print("  → Ikuti instruksi di README untuk data asli dari bi.go.id/hargapangan\n")
        return _generate_simulated_local_prices()

    print(f"► Memuat data PIHPS dari {filepath}...")
    df = pd.read_csv(filepath, parse_dates=["tanggal"], index_col="tanggal")
    print(f"  ✓ {len(df)} baris dimuat")
    return df


def _generate_simulated_local_prices() -> pd.DataFrame:
    """Bangkitkan data simulasi realistis sebagai placeholder sementara."""
    import numpy as np
    dates = pd.date_range(START_DATE, datetime.today(), freq="D")
    rng = np.random.default_rng(42)

    def price_series(base, trend=0.0002, noise=0.01, seasonal_amp=0.05):
        n = len(dates)
        t = np.arange(n)
        seasonal = seasonal_amp * np.sin(2 * np.pi * t / 365)
        noise_arr = rng.normal(0, noise, n)
        return base * np.cumprod(1 + trend + noise_arr + seasonal / n)

    df = pd.DataFrame({
        "Beras"       : price_series(12000, trend=0.0003),
        "Minyak Goreng": price_series(15000, trend=0.0002),
        "Telur Ayam"  : price_series(25000, trend=0.0001, noise=0.015),
        "Cabai Merah" : price_series(35000, trend=0.0000, noise=0.04, seasonal_amp=0.2),
        "Daging Ayam" : price_series(33000, trend=0.0002, noise=0.012),
    }, index=dates)
    df.index.name = "tanggal"
    _save(df, "pihps_harga.csv")
    return df


# ── 3. BMKG cuaca (placeholder) ──────────────────────────────────────────

def fetch_bmkg_weather() -> pd.DataFrame:
    """
    Placeholder data cuaca dari BMKG.
    BMKG menyediakan data via https://dataonline.bmkg.go.id (perlu registrasi).
    Untuk sementara menggunakan data curah hujan simulasi.
    """
    print("► Data BMKG: menggunakan simulasi curah hujan...")
    import numpy as np
    dates = pd.date_range(START_DATE, datetime.today(), freq="D")
    rng = np.random.default_rng(99)
    rainfall = np.abs(rng.normal(5, 8, len(dates)))          # mm/hari
    temperature = 27 + 3 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + rng.normal(0, 1, len(dates))
    df = pd.DataFrame({"curah_hujan_mm": rainfall, "suhu_c": temperature}, index=dates)
    df.index.name = "tanggal"
    _save(df, "bmkg_cuaca.csv")
    return df


# ── 4. Kalender hari besar Indonesia ─────────────────────────────────────

def get_holiday_features() -> pd.DataFrame:
    """Buat fitur biner untuk hari besar nasional (pengaruh permintaan musiman)."""
    dates = pd.date_range(START_DATE, datetime.today(), freq="D")
    df = pd.DataFrame(index=dates)
    df.index.name = "tanggal"

    # Ramadan & Lebaran (approx — sesuaikan manual tiap tahun)
    ramadan_windows = [
        ("2020-04-23", "2020-05-23"), ("2021-04-12", "2021-05-12"),
        ("2022-04-02", "2022-05-01"), ("2023-03-22", "2023-04-20"),
        ("2024-03-11", "2024-04-09"), ("2025-03-01", "2025-03-30"),
        ("2026-02-18", "2026-03-19"),
    ]
    df["is_ramadan"] = 0
    for start, end in ramadan_windows:
        mask = (df.index >= start) & (df.index <= end)
        df.loc[mask, "is_ramadan"] = 1

    df["bulan"] = df.index.month
    df["hari_dalam_minggu"] = df.index.dayofweek
    _save(df, "holiday_features.csv")
    return df


# ── Main ──────────────────────────────────────────────────────────────────

def collect_all_data():
    """Jalankan semua scraper dan kembalikan dict berisi semua DataFrame."""
    print("=" * 50)
    print("  Pengumpulan Data Dimulai")
    print("=" * 50)
    data = {
        "external"  : fetch_yahoo_data(),
        "local"     : load_bapanas_data(),
        "weather"   : fetch_bmkg_weather(),
        "holidays"  : get_holiday_features(),
    }
    print("\n✅ Semua data berhasil dikumpulkan.")
    return data


if __name__ == "__main__":
    collect_all_data()
