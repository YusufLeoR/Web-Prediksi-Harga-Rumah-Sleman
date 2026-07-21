# 🏠 Prediksi Harga Properti Kabupaten Sleman

Aplikasi Streamlit untuk memprediksi harga properti di Kabupaten Sleman menggunakan **XGBoost Regression** yang dibungkus dalam **scikit-learn Pipeline**.

## Struktur Proyek

```
.
├── app.py
├── model_prediksi_harga_sleman_audit.pkl
├── sample_per_kecamatan.csv
├── requirements.txt
└── README.md
```

## Cara Menjalankan

### Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Streamlit Cloud

1. Push semua file ke GitHub repository.
2. Buka [share.streamlit.io](https://share.streamlit.io) dan hubungkan repo.
3. Set `app.py` sebagai entry point.
4. Deploy.

## Fitur Aplikasi

- **Input 7 fitur** dari pengguna (luas tanah, luas bangunan, kamar tidur, kamar mandi, garasi, carport, kecamatan)
- **Auto-fill 16 fitur spasial** berdasarkan kecamatan yang dipilih
- **Hasil prediksi**: estimasi harga, harga per m², dan kategori (Ekonomis / Menengah / Premium)
- **Riwayat prediksi** tersimpan selama sesi dan dapat diunduh sebagai CSV
- **Contoh data** bawaan untuk demo cepat

## Kategori Harga

| Kategori  | Rentang Harga           |
|-----------|-------------------------|
| Ekonomis  | < Rp 500.000.000        |
| Menengah  | Rp 500jt – Rp 1,5 Miliar |
| Premium   | > Rp 1.500.000.000      |
