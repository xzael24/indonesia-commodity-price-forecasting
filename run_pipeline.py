# run_pipeline.py
"""
Script utama — jalankan SATU FILE ini untuk:
  1. Kumpulkan data
  2. Preprocessing
  3. Training semua model
  4. Simpan hasil evaluasi

Jalankan: python run_pipeline.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.scraper    import collect_all_data
from src.preprocess import run_preprocessing
from src.train      import run_training

if __name__ == "__main__":
    print("\n[Pipeline] Prediksi Harga Kebutuhan Pokok")
    print("=" * 50)

    print("\n[STEP 1/3] Pengumpulan Data")
    data = collect_all_data()

    print("\n[STEP 2/3] Preprocessing")
    run_preprocessing(data, target_commodity="Beras")

    print("\n[STEP 3/3] Training Model")
    run_training()

    print("\n[Pipeline] Selesai!")
    print("   Jalankan dashboard: streamlit run app.py")
