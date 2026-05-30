# src/commodity_manager.py
"""
Utility untuk Manajemen Komoditas Dinamis:
1. Ekstraksi nama kategori utama komoditas (menghapus variasi/subkategori).
2. Parsing file Excel/CSV (layout baris PIHPS/Bapanas atau layout kolom waktu).
3. Integrasi data ke pihps_harga.csv dengan penanganan missing values (ffill, bfill).
4. Penghapusan komoditas (kolom dataset, model best pt checkpoints, preprocessed y, hasil results.json).
"""

import os
import re
import json
import glob
import numpy as np
import pandas as pd
from config import DATA_RAW_DIR, DATA_PROCESSED_DIR, MODEL_DIR

def extract_main_commodity(name: str) -> str:
    """
    Ekstrak nama kategori utama komoditas dengan membersihkan angka romawi,
    angka biasa, imbuhan kualitas, ukuran, super, curah, dll.
    Contoh:
      - "I. Bawang Merah" -> "Bawang Merah"
      - "1 Bawang Merah Ukuran Sedang" -> "Bawang Merah"
      - "Bawang Merah Super" -> "Bawang Merah"
      - "Minyak Goreng Curah" -> "Minyak Goreng"
    """
    if not isinstance(name, str):
        name = str(name)
    name = name.strip()
    
    # 1. Bersihkan angka Romawi / angka biasa di awal (misal: "I. Bawang Merah" atau "1. Bawang Merah")
    name = re.sub(r'^[IVXLCDM]+\.?\s+', '', name)
    name = re.sub(r'^\d+\.?\s+', '', name)
    
    # 2. Bersihkan satuan harga (misal: "(Rp)", "(Rp/ kg)", "(Rp/kg)")
    name = re.sub(r'\s*\(\s*Rp\s*/?\s*\w*\s*\)', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*\(\s*Rp\s*\)', '', name, flags=re.IGNORECASE)
    
    # 3. Hapus kata kunci subkategori yang umum (case-insensitive)
    patterns_to_strip = [
        r'\bKualitas\s+(Medium|Premium|Curah|Super|Biasa)\s+(I|II|III|IV|V|1|2|3|4|5)\b',
        r'\bKualitas\s+(I|II|III|IV|V|1|2|3|4|5)\b',
        r'\bKualitas\s+(Medium|Premium|Curah|Super|Biasa)\b',
        r'\bUkuran\s+(Kecil|Sedang|Besar)\b',
        r'\b(Kecil|Sedang|Besar)\b',
        r'\bSuper\b',
        r'\bCurah\b',
        r'\bSegar\b',
        r'\bPremium\b',
        r'\bMedium\b',
        r'\bAntar\s+Kota\b',
        r'\bLokal\b',
    ]
    
    for pattern in patterns_to_strip:
        name = re.sub(pattern, '', name, flags=re.IGNORECASE)
        
    # Bersihkan spasi ganda, tanda hubung di akhir/awal, dan spasi
    name = re.sub(r'\s+', ' ', name)
    name = re.sub(r'\s*-\s*$', '', name)
    name = re.sub(r'^\s*-\s*', '', name)
    name = name.strip()
    
    return name

def parse_uploaded_file(file_path_or_buffer) -> pd.DataFrame:
    """
    Mendeteksi layout file dan memparsingnya menjadi DataFrame yang terindeks 'Date' (DatetimeIndex)
    dengan kolom sebagai nama komoditas yang sudah dibersihkan.
    Mendukung format:
      1. Row-based (PIHPS/Bapanas: Tanggal di kolom, komoditas di baris)
      2. Column-based (Time-series standar: Tanggal di baris/index, komoditas di kolom)
    """
    # 1. Baca file
    if isinstance(file_path_or_buffer, str) and file_path_or_buffer.endswith('.csv'):
        df = pd.read_csv(file_path_or_buffer)
    else:
        df = pd.read_excel(file_path_or_buffer)
        
    # Standarisasi spasi kolom
    df.columns = [str(c).strip() for c in df.columns]
    columns = df.columns.tolist()
    
    # 2. Cek apakah ini Layout Baris (Row-based: dates are columns)
    date_cols = []
    non_date_cols = []
    
    for col in columns:
        cleaned_col = re.sub(r'\s+', '', col)  # Hilangkan spasi: "01/ 01/ 2020" -> "01/01/2020"
        try:
            # Cek pola tanggal standard dd/mm/yyyy atau yyyy-mm-dd
            if (re.match(r'^\d{1,2}[/\-\s]\d{1,2}[/\-\s]\d{4}$', cleaned_col) or 
                re.match(r'^\d{4}[/\-\s]\d{1,2}[/\-\s]\d{1,2}$', cleaned_col)):
                pd.to_datetime(cleaned_col, dayfirst=True)
                date_cols.append(col)
            else:
                non_date_cols.append(col)
        except Exception:
            non_date_cols.append(col)
            
    # Jika kolom tanggal banyak (> 5) dan ada kolom non-tanggal, ini layout baris
    if len(date_cols) > 5 and len(non_date_cols) > 0:
        print("[INFO] Terdeteksi format: ROW-BASED (PIHPS/Bapanas Standard)")
        
        # Cari kolom nama komoditas
        commodity_col = None
        for col in non_date_cols:
            if 'komoditas' in col.lower():
                commodity_col = col
                break
        if commodity_col is None:
            # Fallback ke kolom pertama non-date yang bukan 'No'
            fallback_cols = [c for c in non_date_cols if c.lower() not in ['no', 'no.']]
            commodity_col = fallback_cols[0] if fallback_cols else non_date_cols[0]
            
        # Bentuk ulang (pivot) dataframe agar tanggal menjadi index dan komoditas jadi kolom
        records = []
        for _, row in df.iterrows():
            raw_comm_name = row[commodity_col]
            if pd.isna(raw_comm_name):
                continue
                
            comm_name = extract_main_commodity(raw_comm_name)
            if not comm_name:
                continue
                
            for date_col in date_cols:
                val = row[date_col]
                val_str = str(val).strip()
                
                # Cek data kosong
                if val_str in ['-', '', 'nan', 'NaN', 'None', 'null']:
                    price = np.nan
                else:
                    # Hilangkan koma ribuan dan Rp
                    clean_val = val_str.replace(',', '').replace(' ', '').replace('Rp', '').strip()
                    try:
                        price = float(clean_val)
                    except ValueError:
                        price = np.nan
                        
                # Parse tanggal
                cleaned_date_str = re.sub(r'\s+', '', date_col)
                try:
                    dt = pd.to_datetime(cleaned_date_str, dayfirst=True)
                    records.append({
                        'Date': dt,
                        'Commodity': comm_name,
                        'Price': price
                    })
                except Exception:
                    continue
                    
        df_flat = pd.DataFrame(records)
        if df_flat.empty:
            raise ValueError("Tidak ada data harga valid yang berhasil diparsing dari layout baris.")
            
        # Group by Date & Commodity, hitung rata-rata harian (daily average) untuk subkategori
        df_grouped = df_flat.groupby(['Date', 'Commodity'])['Price'].mean().unstack()
        df_grouped = df_grouped.sort_index()
        return df_grouped

    else:
        print("[INFO] Terdeteksi format: COLUMN-BASED (Time-series Standard)")
        
        # Cari kolom tanggal
        date_col = None
        for col in columns:
            if col.lower() in ['tanggal', 'date', 'waktu', 'time', 'index']:
                date_col = col
                break
        if date_col is None:
            # Cari kolom pertama yang bisa diparsing sebagai tanggal
            for col in columns:
                try:
                    pd.to_datetime(df[col].iloc[:5])
                    date_col = col
                    break
                except Exception:
                    continue
        if date_col is None:
            # Fallback ke kolom pertama
            date_col = columns[0]
            
        # Ubah kolom tanggal ke DatetimeIndex
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        df = df.set_index(date_col)
        df.index.name = 'Date'
        
        # Bersihkan nama kolom (nama komoditas)
        new_columns = {}
        for col in df.columns:
            cleaned_col = extract_main_commodity(col)
            if cleaned_col:
                new_columns[col] = cleaned_col
                
        df = df[list(new_columns.keys())]
        df = df.rename(columns=new_columns)
        
        # Bersihkan data harga
        for col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '', regex=False).str.replace(' ', '', regex=False).str.replace('Rp', '', regex=False).str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Gabungkan kolom duplikat dengan rata-rata, gabungkan tanggal duplikat dengan rata-rata
        df = df.T.groupby(level=0).mean().T
        df = df.groupby(df.index).mean()
        df = df.sort_index()
        return df

def integrate_into_dataset(df_new: pd.DataFrame) -> dict:
    """
    Integrasikan DataFrame baru (hasil parsing) ke dataset utama data/raw/pihps_harga.csv.
    Melakukan reindexing tanggal, averaging jika ada overlap, dan imputasi ffill().bfill().
    Mengembalikan ringkasan status operasi integrasi.
    """
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    pihps_path = os.path.join(DATA_RAW_DIR, "pihps_harga.csv")
    
    if os.path.exists(pihps_path):
        df_main = pd.read_csv(pihps_path, index_col=0, parse_dates=True)
    else:
        # Jika file belum ada, buat baru
        df_main = pd.DataFrame(index=df_new.index)
        
    df_main.index.name = 'Date'
    
    # Ringkasan perubahan
    added_cols = []
    updated_cols = []
    
    for col in df_new.columns:
        if col in df_main.columns:
            updated_cols.append(col)
        else:
            added_cols.append(col)
            
    # Gabungkan dengan concat luar (outer join) di level index
    df_combined = pd.concat([df_main, df_new], axis=1)
    
    # Kelompokkan kolom yang memiliki nama sama (misal ada irisan) dan ambil first non-null
    df_combined = df_combined.T.groupby(level=0).first().T
    
    # Urutkan berdasarkan tanggal
    df_combined = df_combined.sort_index()
    
    # Lakukan ffill dan bfill untuk mengisi gap nilai kosong
    df_combined = df_combined.ffill().bfill()
    
    # Simpan kembali ke pihps_harga.csv
    df_combined.to_csv(pihps_path)
    print(f"[OK] Integrasi sukses. Ditambahkan: {added_cols} | Diupdate: {updated_cols}")
    
    return {
        "added": added_cols,
        "updated": updated_cols,
        "start_date": str(df_combined.index.min().date()),
        "end_date": str(df_combined.index.max().date()),
        "total_rows": len(df_combined)
    }

def delete_commodity(commodity_name: str) -> bool:
    """
    Hapus komoditas tertentu dari sistem secara bersih:
    1. Kolom dari pihps_harga.csv
    2. File preprocessed data terkait komoditas
    3. Model checkpoint (*_commodity_best.pt)
    4. Data hasil evaluasi di models/results.json
    """
    # 1. Hapus dari pihps_harga.csv
    pihps_path = os.path.join(DATA_RAW_DIR, "pihps_harga.csv")
    if os.path.exists(pihps_path):
        df_main = pd.read_csv(pihps_path, index_col=0, parse_dates=True)
        if commodity_name in df_main.columns:
            df_main = df_main.drop(columns=[commodity_name])
            df_main.to_csv(pihps_path)
            print(f"[OK] Kolom '{commodity_name}' dihapus dari {pihps_path}")
        else:
            print(f"[WARNING] Kolom '{commodity_name}' tidak ditemukan di pihps_harga.csv")
            
    # 2. Hapus file processed y_*.npy
    y_patterns = [
        f"y_train_{commodity_name}.npy",
        f"y_val_{commodity_name}.npy",
        f"y_test_{commodity_name}.npy"
    ]
    for pattern in y_patterns:
        file_path = os.path.join(DATA_PROCESSED_DIR, pattern)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"[OK] File processed data '{file_path}' dihapus")
            
    # 3. Hapus checkpoints model (*_{commodity_name}_best.pt)
    model_patterns = glob.glob(os.path.join(MODEL_DIR, f"*_{commodity_name}_best.pt"))
    for file_path in model_patterns:
        os.remove(file_path)
        print(f"[OK] Model best checkpoint '{file_path}' dihapus")
        
    # 4. Hapus dari results.json
    results_path = os.path.join(MODEL_DIR, "results.json")
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r') as f:
                results = json.load(f)
            if commodity_name in results:
                del results[commodity_name]
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"[OK] Entri '{commodity_name}' dihapus dari results.json")
        except Exception as e:
            print(f"[WARNING] Gagal memperbarui results.json: {e}")
            
    return True
