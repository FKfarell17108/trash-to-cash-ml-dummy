# trash-to-cash-ml-dummy

Dummy machine learning model for the Trash to Cash project. This repository contains a preliminary SVM-based classifier for identifying organic and inorganic waste, including mixed waste detection with percentage estimation. Intended for backend integration and testing purposes. Dataset sourced from Roboflow (CC BY 4.0).

## Isi Folder
- `trash_classifier_model.pkl`: Model SVM yang sudah dilatih (diunggah via Drive: https://drive.google.com/file/d/1RCSbVDj6brzm2aKvQANTinoG71MbgYnA/view?usp=sharing).
- `inference.py`: Skrip inferensi untuk prediksi gambar.
- `requirements.txt`: Daftar dependensi Python.
- `README.md`: Dokumentasi ini.
- `images/`: Folder untuk menyimpan gambar yang akan diuji.
- `notebook/klasifikasi_sampah.ipynb`: Notebook Colab untuk pelatihan dan pengembangan.

## Cara Penggunaan
1. **Instal Dependensi:**
   ```bash
   pip install -r requirements.txt
   
  Pastikan Git LFS terinstal jika mengunduh `trash_classifier_model.pkl` dari repo:
   ```bash
   git lfs install
```

2. **Siapkan Gambar**:
- Tambahkan gambar ke folder images/ (misalnya 1.jpg, 2.jpeg.jpg).
- Gambar minimal 128x128 piksel untuk deteksi campuran.

3. **Jalankan Inferensi:**
   ```bash
   python inference.py
- Output akan menampilkan hasil klasifikasi biner dan campuran untuk setiap gambar.
- Gambar yang ditandai disimpan sebagai marked_<nama_file>.jpg.

## Spesifikasi Input
- Klasifikasi Biner: Gambar diresize ke 64x64 piksel dan diflatten menjadi vektor 12,288 elemen (RGB).
- Klasifikasi Campuran: Gambar minimal 128x128 piksel (grid 2x2). Hasil berupa persentase organik/anorganik dan gambar ditandai.
- Format Gambar: .jpg, .jpeg, atau .png.

## Struktur Output
- Biner: {"prediction": "Organik"} atau {"prediction": "Anorganik"}.
- Campuran:
  ```bash
   {
  "organik_percent": 75.0,
  "anorganik_percent": 25.0,
  "grid_predictions": ["Organik", "Organik", "Organik", "Anorganik"]
  }
  ```
  Ditambah file gambar marked_<nama_file>.jpg.

## Integrasi untuk Backend
- Endpoint API:
  - `/predict`: Untuk klasifikasi biner (input: path gambar, output: JSON).
  - `/predict_mixed`: Untuk klasifikasi campuran (input: path gambar, output: JSON + file gambar ditandai).
- Gunakan Flask/FastAPI untuk membungkus `predict_image()` dan `predict_mixed_waste()` di `inference.py`.

## Lisensi
MIT License

Copyright (c) 2025 CC25-SF039

Permission is hereby granted, free of charge, to any person obtaining a copy of this software...

See the LICENSE file for details. Dataset sourced from Roboflow under CC BY 4.0.

## Atribusi
- Dataset: Klasifikasi Sampah Organik dan Anorganik by Skripsi Aji (CC BY 4.0).
- Link: https://universe.roboflow.com/skripsi-aji/klasifikasi-sampah-organik-dan-anorganik/dataset/35

## Catatan
- Notebook: https://colab.research.google.com/drive/1gK8RAOlisLqzVyOgGLH4jzO0M7eKBas_
- File Besar: File trash_classifier_model.pkl melebihi 100 MB, file ini tidak ada di repo langsung. Unduh dari Link Google Drive: https://drive.google.com/file/d/1RCSbVDj6brzm2aKvQANTinoG71MbgYnA/view?usp=sharing
- Ukuran Model: Model dummy dilatih dengan subset data 5000 sample. Untuk versi produksi, latih ulang dengan dataset penuh.
- Kontak: Hubungi farellkurniawan17108@gmail.com untuk pertanyaan atau bantuan teknis.
