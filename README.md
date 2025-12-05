Bayesian Network – Sachs Dataset Analysis

Proyek ini berisi implementasi lengkap untuk mempelajari Conditional Probability Tables (CPTs), melakukan inferensi probabilistik, visualisasi jaringan Bayesian, serta evaluasi model menggunakan dataset Sachs (jaringan regulasi sinyal sel). Program mencakup proses mulai dari data generation, exploratory analysis, parameter learning, CPT visualization, hingga posterior inference dengan evidence tertentu.

Struktur Proyek

```
project/
│
├── Bayesian.py                   # Modul berisi fungsi CPT learning, inference, scoring, visualisasi
├── data/
│   └── sachs.bif/                # File BIF dan output CSV
│       ├── sachs.bif
│       ├── sachs.csv
│       └── sachs_cpts1.csv
│
├── data/generator.py             # Modul generator dataset dari file BIF
│
├── notebook/
│   └── main_experiment.ipynb     # Notebook utama untuk menjalankan semua eksperimen
│
├── requirements.txt              # Daftar dependensi
└── README.md
```

Fitur Utama

Data Generation
Menggunakan `pgmpy` untuk membaca file `.bif` lalu membangkitkan ribuan sampel observasi menggunakan forward sampling.

Exploratory Data Analysis (EDA)
Visualisasi distribusi state untuk setiap node, pengecekan konsistensi variabel, dan pembuatan heatmap korelasi (Spearman).

Parameter Learning (MLE + Laplace Smoothing)
Mengestimasi CPT dari data dengan menghitung frekuensi kondisi setiap kombinasi parent–child.

Visualisasi Bayesian Network
Tampilan graf struktur jaringan + tabel CPT setiap node.

Model Evaluation

- Log-Likelihood
- Per-Node Prediction Accuracy
- BIC Score
- 5-Fold Cross Validation

Posterior Inference
Menghitung distribusi posterior tiap node berdasarkan evidence tertentu, disajikan dalam bar plot bergaya Excel.

Cara Menjalankan Program

Install dependencies

```
pip install -r requirements.txt
```

Generate dataset Sachs (jika belum tersedia)
Jalankan fungsi:

```
from data.generator import sample
sample("data/sachs.bif/sachs.bif", 5500, "data/sachs.bif/sachs.csv")
```

Buka notebook utama

`notebook/main_experiment.ipynb`

Jalankan semua sel secara berurutan:

- Import library
- Generate & load dataset
- EDA
- Parameter learning (CPT)
- Visualisasi BN & CPT
- Evaluasi model
- Inferensi dan visualisasi posterior

Hasil utama

- CPT model tersimpan di: `data/sachs.bif/sachs_cpts1.csv`
- Gambar Bayesian Network
- Tabel CPT tiap node
- Hasil evaluasi log-likelihood, BIC, akurasi
- Grafik posterior seluruh node berdasarkan evidence

requirements.txt

- numpy
- pandas
- matplotlib
- seaborn
- pgmpy
- networkx
- scikit-learn
