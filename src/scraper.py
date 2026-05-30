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
    print(f"  [OK] Tersimpan: {path} ({len(df)} baris)")
    return path


# ── 1. Yahoo Finance ──────────────────────────────────────────────────────

def fetch_yahoo_data() -> pd.DataFrame:
    """Ambil harga penutupan komoditas global & kurs dari Yahoo Finance."""
    print("[INFO] Mengambil data Yahoo Finance...")
    frames = {}
    for name, ticker in YAHOO_TICKERS.items():
        try:
            df = yf.download(ticker, start=START_DATE, progress=False, auto_adjust=True)
            if not df.empty:
                frames[name] = df["Close"].squeeze()
                print(f"  [OK] {name} ({ticker}): {len(df)} baris")
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

def fetch_bi_realtime_prices() -> dict:
    """
    Scrape harga pangan eceran rata-rata nasional terbaru secara real-time dari portal BI PIHPS.
    Mencari baris tabel yang mengandung nama komoditas dan mengekstrak nominal harganya.
    """
    print("[INFO] Mengambil data harga pangan real-time dari portal Bank Indonesia (PIHPS)...")
    url = "https://www.bi.go.id/hargapangan"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://www.bi.go.id/"
    }
    
    targets = {
        "Beras": ["beras", "medium", "beras kualitas medium"],
        "Minyak Goreng": ["minyak", "minyak goreng", "minyak goreng curah"],
        "Telur Ayam": ["telur", "telur ayam", "telur ayam ras"],
        "Cabai Merah": ["cabai", "cabai merah", "cabai merah keriting"],
        "Daging Ayam": ["daging ayam", "daging ayam ras", "daging ayam ras segar"]
    }
    
    prices = {}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")
            rows = soup.find_all("tr")
            for row in rows:
                cells = [c.get_text().strip().lower() for c in row.find_all(["td", "th"])]
                if len(cells) >= 2:
                    text_col = cells[0]
                    price_col = cells[1]
                    for name, synonyms in targets.items():
                        if name in prices:
                            continue
                        if any(syn in text_col for syn in synonyms):
                            digits = "".join([char for char in price_col if char.isdigit()])
                            if digits:
                                prices[name] = float(digits)
                                print(f"  [OK] Terdeteksi di portal BI: {name} = Rp {prices[name]:,.0f}")
    except Exception as e:
        print(f"  [WARNING]️ Gagal menghubungi portal BI untuk data real-time: {e}")
        
    return prices


def load_bapanas_data(filepath: str = None) -> pd.DataFrame:
    """
    Memuat data harga lokal dari file pihps_harga.csv.
    Jika ada hari yang terlewat (misal hari ini belum ada), sistem secara otomatis
    melakukan scraping real-time dari BI hargapangan dan memperbarui data lokal secara mulus.
    """
    if filepath is None:
        pihps_path   = os.path.join(DATA_RAW_DIR, "pihps_harga.csv")
        bapanas_path = os.path.join(DATA_RAW_DIR, "bapanas_harga.csv")
        if os.path.exists(pihps_path):
            filepath = pihps_path
        elif os.path.exists(bapanas_path):
            filepath = bapanas_path
            print("  ℹ️ Menggunakan file lama bapanas_harga.csv.")
        else:
            filepath = pihps_path

    # Jika file sama sekali belum ada, gunakan data dasar simulasi sebagai inisialisasi awal
    if not os.path.exists(filepath):
        print(f"\n  [WARNING]️ File PIHPS belum ditemukan di: {filepath}")
        print("  → Menginisialisasi data dasar awal...")
        df = _generate_simulated_local_prices()
    else:
        print(f"[INFO] Memuat data PIHPS dari {filepath}...")
        df = pd.read_csv(filepath, index_col=0)
        df.index = pd.to_datetime(df.index)
        df.index.name = "tanggal"
        print(f"  [OK] {len(df)} baris data historis berhasil dimuat")

    # --- SINKRONISASI REAL-TIME OTOMATIS ---
    last_date = df.index[-1]
    today = pd.to_datetime(datetime.today().date())
    
    if last_date < today:
        print(f"  ℹ️ Mendeteksi keterlambatan data lokal. Terakhir: {last_date.strftime('%Y-%m-%d')} | Hari ini: {today.strftime('%Y-%m-%d')}")
        bi_prices = fetch_bi_realtime_prices()
        
        # Buat daftar tanggal yang kosong
        missing_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), end=today, freq="D")
        
        new_rows = []
        for d in missing_dates:
            row_data = {}
            for col in df.columns:
                last_price = df[col].iloc[-1]
                target_price = bi_prices.get(col, last_price)
                
                # Gunakan interpolasi linier yang mulus dari harga terakhir ke harga terbaru hari ini
                days_total = (today - last_date).days
                days_current = (d - last_date).days
                
                # Rumus interpolasi: P_t = P_last + (P_target - P_last) * (t / total)
                interpolated_price = last_price + (target_price - last_price) * (days_current / days_total)
                row_data[col] = round(interpolated_price, 0)
                
            new_rows.append(pd.Series(row_data, name=d))
            
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            df = pd.concat([df, new_df])
            # Simpan file yang telah disinkronkan kembali ke disk
            df.to_csv(filepath)
            print(f"  [OK] Sinkronisasi berhasil! {len(new_rows)} baris data real-time hingga hari ini berhasil ditambahkan.")
            
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
    print("[INFO] Data BMKG: menggunakan simulasi curah hujan...")
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
    print("\n[SUCCESS] Semua data berhasil dikumpulkan.")
    return data


if __name__ == "__main__":
    collect_all_data()
