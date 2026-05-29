# 🔮 Prediksi Harga Kebutuhan Pokok Indonesia

Sistem prediksi harga bahan pokok berbasis **Deep Learning Multivariate Time Series**
dengan integrasi faktor ekonomi global (kurs rupiah, harga komoditas dunia, cuaca).

---

## 📁 Struktur Project

```
price-prediction/
├── app.py                  # Dashboard Streamlit
├── run_pipeline.py         # Script utama (jalankan ini)
├── config.py               # Semua konfigurasi
├── requirements.txt
├── src/
│   ├── scraper.py          # Ambil data otomatis
│   ├── preprocess.py       # Preprocessing & feature engineering
│   ├── model.py            # LSTM, GRU, TFT
│   └── train.py            # Training & evaluasi
├── data/
│   ├── raw/                # Data mentah (auto-generated)
│   └── processed/          # Data siap pakai (auto-generated)
└── models/                 # Model tersimpan (auto-generated)
```

---

## ⚙️ Setup (Ikuti Urutan Ini!)

### Step 1 — Install Python & dependencies

Pastikan Python 3.9+ sudah terinstall, lalu:

```bash
pip install -r requirements.txt
```

> Jika error PyTorch, install manual sesuai OS di https://pytorch.org/get-started/locally/

---

### Step 2 — Download data PIHPS Bank Indonesia (MANUAL — WAJIB)

> ⚠️ Bagian ini **harus dilakukan manual** karena tidak ada API publik.
> Gunakan **PIHPS Bank Indonesia** (aktif & tidak maintenance) sebagai sumber data harga lokal.

1. Buka browser → **https://www.bi.go.id/hargapangan**
2. Pilih menu **"Tabel Harga Berdasarkan Daerah"**
3. Isi filter berikut:
   - **Komoditas:** pilih satu per satu (Beras, Minyak Goreng, Telur Ayam, Cabai Merah, Daging Ayam)
   - **Jenis Pasar:** Pasar Tradisional
   - **Provinsi:** Nasional (atau pilih provinsi tertentu)
   - **Tanggal Mulai:** 01/01/2020
   - **Tanggal Selesai:** hari ini
4. Klik **"Lihat Laporan"**
5. Klik tombol **Export → Excel atau CSV**
6. Ulangi langkah 3–5 untuk setiap komoditas
7. **Gabungkan semua kolom** ke satu file dengan format berikut:
8. **Rename file menjadi** `pihps_harga.csv`
9. **Simpan di** `data/raw/pihps_harga.csv`

Format kolom yang diharapkan (pastikan header sama persis):

```
tanggal,Beras,Minyak Goreng,Telur Ayam,Cabai Merah,Daging Ayam
2020-01-01,12000,14000,24000,32000,31000
...
```

> 💡 **Kenapa PIHPS Bank Indonesia?**
> Data harian resmi dari 82 kota/kabupaten seluruh Indonesia, disurvei setiap hari kerja oleh Bank Indonesia. Lebih lengkap dan stabil dibanding sumber lain.

> 💡 Jika format CSV berbeda, sesuaikan nama kolom di `config.py` → bagian `COMMODITIES`

---

### Step 3 — Jalankan pipeline

```bash
python run_pipeline.py
```

Ini akan otomatis:

- ✅ Mengambil data Yahoo Finance (kurs, komoditas global)
- ✅ Memuat data PIHPS
- ✅ Preprocessing + feature engineering
- ✅ Training LSTM, GRU, dan TFT
- ✅ Menyimpan model terbaik

Estimasi waktu: **15–30 menit** (tergantung CPU/GPU)

---

### Step 4 — Jalankan dashboard

```bash
streamlit run app.py
```

Buka browser → http://localhost:8501

---

## 🤖 Model

| Model          | Deskripsi                   | Kapan Digunakan               |
| -------------- | --------------------------- | ----------------------------- |
| **LSTM** | Long Short-Term Memory      | Baseline klasik               |
| **GRU**  | Gated Recurrent Unit        | Baseline ringan               |
| **TFT**  | Temporal Fusion Transformer | Main model, akurasi tertinggi |

---

## 📊 Metrik Evaluasi

| Metrik         | Keterangan                                                       |
| -------------- | ---------------------------------------------------------------- |
| **MAE**  | Mean Absolute Error — rata-rata selisih prediksi vs aktual (Rp) |
| **RMSE** | Root Mean Square Error — penalti lebih besar untuk error besar  |
| **MAPE** | Mean Absolute Percentage Error — error dalam persen (%)         |

Target: MAPE < 5% untuk komoditas stabil (beras, telur)

---

## ⚡ Tips Peningkatan Akurasi

1. **Tambah data lebih lama** — ubah `START_DATE` di `config.py` ke 2018
2. **Tuning hyperparameter** — edit `HIDDEN_SIZE`, `NUM_LAYERS`, `EPOCHS` di `config.py`
3. **Tambah fitur** — edit fungsi `add_features()` di `src/preprocess.py`
4. **Gunakan GPU** — PyTorch otomatis detect CUDA jika tersedia

---

## 🏆 Untuk Kompetisi

Sebelum submit kompetisi, pastikan:

- [ ] Data asli dari Bapanas (bukan simulasi)
- [ ] Dokumentasi metodologi tersedia (lihat notebook/)
- [ ] MAPE < 5% pada test set
- [ ] Dashboard berjalan lancar
- [ ] README ini sudah diperbarui dengan hasil aktual

---

## 📄 Teknologi

- **Python 3.9+**
- **PyTorch** — deep learning framework
- **Streamlit** — dashboard interaktif
- **Plotly** — visualisasi
- **yfinance** — data komoditas global
- **scikit-learn** — preprocessing
