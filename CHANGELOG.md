# Changelog

Semua perubahan penting pada proyek ini akan didokumentasikan di file ini. Format ini mengacu pada prinsip Keep a Changelog.

## [1.1.0] - 2026-05-30

### Ditambahkan
- **Manajemen Komoditas Dinamis (`src/commodity_manager.py`)**: Implementasi modul penanganan komoditas baru. Mendukung parsing otomatis dokumen Excel/CSV dengan struktur row-based (format BI PIHPS) maupun column-based, penyelarasan indeks tanggal, penggabungan ke dataset utama dengan interpolasi linier, serta penghapusan komoditas secara bersih dari dataset, bobot model, dan hasil evaluasi.
- **Arsitektur Model N-Linear (`src/model.py`)**: Integrasi model NLinear (Normalized Linear) sebagai opsi model peramalan runtun waktu berkinerja tinggi namun efisien komputasi, guna mengatasi pergeseran distribusi temporal (*distribution shifts*).
- **Sinkronisasi Data Real-time Bank Indonesia (`src/scraper.py`)**: Modul web scraper berbasis BeautifulSoup untuk menarik harga eceran nasional terbaru dari portal resmi PIHPS Bank Indonesia secara otomatis saat mendeteksi keterlambatan data lokal.
- **Skrip Penjadwal Otomatis (`run_scheduler.py`)**: Daemon latar belakang yang mengeksekusi pipeline pelatihan ulang model secara otomatis setiap hari pada pukul 23:00 WIB, dengan output log terarah ke `models/scheduler.log`.
- **Panel Administrasi Dashboard (`app.py`)**: Konsol visual baru pada dashboard Streamlit untuk mengunggah berkas data baru, memantau log pelatihan model secara real-time via antarmuka web, serta menghapus data komoditas tertentu.
- **Fitur Lag Ekonomi & Cuaca (`src/preprocess.py`)**: Penambahan lag fitur makroekonomi (USD/IDR, harga minyak mentah, komoditas pangan global) dan cuaca (curah hujan, suhu) untuk 7, 14, dan 30 hari untuk menangkap efek transmisi harga yang tertunda.

### Diubah
- **Fungsi Loss Pelatihan (`src/model.py`)**: Transisi fungsi loss dari Mean Squared Error (MSE Loss) ke Huber Loss (`SmoothL1Loss` dengan $\beta=0.1$) untuk meningkatkan ketahanan model terhadap lonjakan harga (*price spikes*) ekstrem.
- **Konfigurasi Komoditas Dinamis (`config.py`)**: Variabel list `COMMODITIES` kini dibaca secara dinamis dari kolom berkas data lokal `pihps_harga.csv` alih-alih didefinisikan secara statis.
- **Refaktorisasi Modul Pelatihan (`src/train.py`)**: Pemisahan alur pelatihan menjadi fungsi modular per komoditas (`train_single_commodity`) serta penyimpanan hasil evaluasi ke `results.json` secara incremental dan aman dari kegagalan proses tengah jalan.
- **Metode Imputasi Data (`src/preprocess.py`)**: Pembaruan sintaks interpolasi pandas dari `fillna(method="ffill")` ke metode standar `.ffill().bfill()` yang kompatibel dengan versi terbaru.
- **Pembersihan Logika Penulisan Console**: Standarisasi format penulisan output konsol menggunakan tag status formal (`[INFO]`, `[OK]`, `[SUCCESS]`) untuk menggantikan simbol grafis non-standar.
