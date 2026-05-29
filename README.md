# Indonesia Commodity Price Forecasting System

Sistem prediksi harga bahan pokok berbasis deep learning multivariate time series. Sistem ini mengintegrasikan harga pangan domestik (PIHPS Bank Indonesia) dengan indikator makroekonomi global (kurs USD/IDR, harga komoditas dunia) dan data cuaca lokal untuk menghasilkan estimasi harga komoditas pangan yang akurat menggunakan arsitektur LSTM, GRU, dan Temporal Fusion Transformer (TFT).

## Struktur Repositori

```text
price-prediction/
├── app.py                  # Dashboard visualisasi interaktif (Streamlit)
├── run_pipeline.py         # Orkestrator pipeline (scraping, preprocessing, training)
├── config.py               # Konfigurasi parameter model dan variabel sistem
├── requirements.txt        # Daftar dependensi Python
├── src/
│   ├── scraper.py          # Modul ekstraksi data makroekonomi dan cuaca (yfinance/API)
│   ├── preprocess.py       # Modul rekayasa fitur (feature engineering) dan pembersihan data
│   ├── model.py            # Definisi arsitektur model (LSTM, GRU, TFT)
│   └── train.py            # Modul pelatihan, evaluasi, dan penyimpanan model
├── data/
│   ├── raw/                # Direktori penyimpanan data mentah
│   └── processed/          # Direktori penyimpanan data hasil rekayasa fitur
└── models/                 # Direktori penyimpanan bobot model terbaik (.pt)
```

## Prasyarat Sistem

- Python 3.9 atau versi yang lebih baru
- PyTorch 2.0+ (disarankan menggunakan CUDA untuk pelatihan model TFT)
- Akses internet untuk ekstraksi data global secara real-time

## Instalasi dan Setup Lingkungan

1. Buat dan aktifkan virtual environment Python:
   ```bash
   python -m venv .venv
   # Untuk Windows (PowerShell):
   .\.venv\Scripts\Activate.ps1
   # Untuk Linux/macOS:
   source .venv/bin/activate
   ```

2. Instal seluruh dependensi yang diperlukan:
   ```bash
   pip install -r requirements.txt
   ```
   *Catatan: Apabila terdapat kendala instalasi PyTorch, silakan merujuk pada dokumentasi resmi [PyTorch](https://pytorch.org/get-started/locally/) untuk menyesuaikan dengan CUDA toolkit pada perangkat Anda.*

## Ingesti Data Domestik (PIHPS Bank Indonesia)

Data harga domestik diakses secara manual karena tidak adanya API publik resmi yang stabil dari Bank Indonesia.

1. Akses portal **PIHPS Bank Indonesia** di [https://www.bi.go.id/hargapangan](https://www.bi.go.id/hargapangan).
2. Navigasi ke menu **Tabel Harga Berdasarkan Daerah**.
3. Ekspor data harian dari **01/01/2020** hingga tanggal saat ini untuk komoditas berikut:
   - Beras
   - Minyak Goreng
   - Telur Ayam
   - Cabai Merah
   - Daging Ayam
4. Pastikan data diekspor menggunakan opsi **Pasar Tradisional** dengan cakupan **Nasional**.
5. Gabungkan hasil ekspor menjadi satu file CSV terpadu dengan nama `pihps_harga.csv` dan simpan pada direktori `data/raw/`.
6. Struktur header kolom pada `pihps_harga.csv` harus mengikuti format berikut:
   ```text
   tanggal,Beras,Minyak Goreng,Telur Ayam,Cabai Merah,Daging Ayam
   2020-01-01,12000,14000,24000,32000,31000
   ```

## Pipeline Eksekusi Sistem

Sistem dirancang dengan pipeline modular yang dapat dioperasikan secara end-to-end melalui satu skrip orkestrasi:

```bash
python run_pipeline.py
```

Skrip di atas akan mengeksekusi tahapan berikut secara berurutan:
1. **Data Scraping**: Mengambil data kurs USD/IDR, indeks harga minyak nabati, dan komoditas global pendukung lainnya secara otomatis dari Yahoo Finance.
2. **Preprocessing**: Membersihkan data, menangani *missing values* melalui metode interpolasi, menggabungkan dataset domestik dan global, serta melakukan standardisasi skala fitur (*scaling*).
3. **Feature Engineering**: Membuat fitur lag (data historis), moving average, dan variabel kalender (tren temporal, hari libur nasional).
4. **Model Training & Evaluation**: Melatih model LSTM, GRU, dan TFT menggunakan parameter yang dikonfigurasi pada `config.py`, melakukan validasi, lalu mengevaluasi kinerja masing-masing model menggunakan metrik MAE, RMSE, dan MAPE.
5. **Artifact Saving**: Menyimpan bobot model terbaik berdasarkan nilai loss validasi terendah ke dalam direktori `models/`.

## Pengoperasian Dashboard Analisis

Setelah pipeline selesai dijalankan dan model tersimpan, jalankan dashboard interaktif Streamlit untuk visualisasi dan inferensi prediksi:

```bash
streamlit run app.py
```
Aplikasi secara otomatis dapat diakses melalui browser pada alamat default `http://localhost:8501`.

## Metodologi Pemodelan

Akurasi sistem dievaluasi secara komparatif menggunakan tiga arsitektur recurrent dan attention-based neural networks:

- **Long Short-Term Memory (LSTM)**: Digunakan sebagai baseline model untuk menangkap dependensi sekuensial jangka panjang pada pola harga komoditas pangan.
- **Gated Recurrent Unit (GRU)**: Arsitektur recurrent yang lebih efisien secara komputasi dengan performa yang kompetitif dibanding LSTM pada data berukuran menengah.
- **Temporal Fusion Transformer (TFT)**: Model utama berbasis attention mechanism yang dirancang khusus untuk peramalan multi-horizon time series. Keunggulan TFT adalah kemampuannya memisahkan fitur statis, fitur temporal yang diketahui (seperti hari libur), dan fitur temporal yang tidak diketahui (seperti harga komoditas global) serta memberikan interpretabilitas melalui mekanisme atensi.

### Metrik Evaluasi Kinerja
Setiap model dinilai berdasarkan metrik standard industri:
- **Mean Absolute Error (MAE)**: Menunjukkan rata-rata absolut selisih prediksi dengan nilai riil dalam satuan Rupiah (IDR).
- **Root Mean Square Error (RMSE)**: Memberikan penalti lebih tinggi pada selisih prediksi yang besar untuk memastikan kestabilan model.
- **Mean Absolute Percentage Error (MAPE)**: Mengukur akurasi relatif dalam persentase, membantu perbandingan performa antar komoditas dengan skala harga yang berbeda.
