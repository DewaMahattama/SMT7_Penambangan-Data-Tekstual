# Review Analyzer: ABSA dengan LDA dan BERT

Analisis sentimen berbasis aspek pada ulasan aplikasi produktivitas di Google Play.
Aspek diekstraksi dengan LDA, sentimen (negatif, netral, positif) diklasifikasikan dengan BERT.

## Hasil pada data uji
Accuracy 70,29% · Macro-F1 0,68 (3 kelas, 1.013 data uji)

## Menjalankan aplikasi
1. Gunakan Python 3.11 atau 3.12.
2. Buat lingkungan dan pasang dependensi:
  python -m venv venv
  venv\Scripts\activate
  pip install -r requirements.txt
3. Siapkan folder `artifacts/` (lihat `artifacts/README.md`).
4. Jalankan:
  python -m streamlit run app.py


## Struktur
- `app.py`: aplikasi Streamlit
- `notebooks/`: notebook pelatihan (Google Colab)
- `data/contoh_ulasan.csv`: contoh berkas untuk tab Unggah CSV

## Dataset
Sentiment Analysis Dataset: Google Play App Reviews (Kaggle, farhaouimouhamed).
