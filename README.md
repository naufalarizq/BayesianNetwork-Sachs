<a name="top"></a>

<h1 align="center">🧠 Bayesian Network – Sachs Dataset Analysis</h1>

| [Sekilas Tentang](#sekilas-tentang) | [Instalasi](#instalasi) | [Konfigurasi](#konfigurasi) | [Cara Pemakaian](#cara-pemakaian) | [Pembahasan](#pembahasan) | [Referensi](#referensi) |
| :---------------------------------: | :---------------------: | :-------------------------: | :-------------------------------: | :-----------------------: | :---------------------: |

[`^ Kembali ke atas ^`](#top)

---

# Sekilas Tentang

**Bayesian Network – Sachs Dataset Analysis** adalah sistem probabilistik berbasis machine learning yang mengimplementasikan Bayesian Network from scratch untuk menganalisis jaringan regulasi sinyal protein dalam sel. Proyek ini dirancang untuk memahami hubungan kausal antar protein menggunakan Conditional Probability Tables (CPTs), inferensi probabilistik, dan evaluasi model yang komprehensif.

## Latar Belakang Masalah

Dalam bidang bioinformatika dan analisis jaringan biologis, peneliti menghadapi tantangan kompleks:

- **Kompleksitas Jaringan Protein**: Interaksi antar protein dalam sel sangat kompleks dan sulit diprediksi
- **Ketidakpastian Data Biologis**: Data observasi mengandung noise dan variabilitas tinggi
- **Inferensi Kausal**: Sulitnya menentukan hubungan sebab-akibat antar molekul biologis
- **Prediksi Perilaku Sel**: Memahami bagaimana perubahan satu protein mempengaruhi protein lain

## Solusi Bayesian Network

Bayesian Network menawarkan pendekatan probabilistik yang powerful dengan:

1. **Directed Acyclic Graph (DAG)** - Representasi visual hubungan kausal:

   - Node = Variabel/Protein (PKC, PKA, Raf, Mek, Erk, dll.)
   - Edge = Dependency/Pengaruh kausal
   - No cycles = Struktur hierarkis yang jelas

2. **Conditional Probability Tables (CPTs)** - Quantifikasi probabilitas:

   - P(Node | Parents) untuk setiap node
   - Maximum Likelihood Estimation (MLE) dari data
   - Laplace Smoothing untuk menghindari zero probability

3. **Probabilistic Inference** - Reasoning under uncertainty:

   - Posterior probability: P(Query | Evidence)
   - Exact inference via enumeration
   - Prediksi state protein berdasarkan observasi

4. **Model Evaluation** - Validasi komprehensif:
   - Log-Likelihood untuk goodness of fit
   - BIC Score untuk model complexity trade-off
   - Per-Node Prediction Accuracy
   - K-Fold Cross Validation untuk generalization

## Fitur Utama

✅ **Data Generation Pipeline** - Generate synthetic data dari BIF file menggunakan pgmpy  
✅ **Exploratory Data Analysis** - Visualisasi distribusi, correlation heatmap, consistency check  
✅ **Parameter Learning** - MLE + Laplace Smoothing untuk CPT estimation  
✅ **Network Visualization** - Graph structure dengan CPT tables terintegrasi  
✅ **Probabilistic Inference** - Exact inference untuk posterior computation  
✅ **Model Evaluation** - Log-Likelihood, BIC, Accuracy, Cross-Validation  
✅ **Interactive Analysis** - Jupyter Notebook dengan visualisasi interaktif

## Tech Stack

| Komponen                    | Tools/Library                        |
| --------------------------- | ------------------------------------ |
| **Language**                | Python 3.8+                          |
| **Data Processing**         | Pandas, NumPy                        |
| **Visualization**           | Matplotlib, Seaborn                  |
| **Network Analysis**        | NetworkX                             |
| **Probabilistic Modeling**  | Custom Implementation (from scratch) |
| **Data Generation**         | pgmpy (BIF reading & sampling)       |
| **Model Evaluation**        | Scikit-learn (KFold)                 |
| **Interactive Environment** | Jupyter Notebook                     |

## Dataset: Sachs et al. (2005)

**Sachs Dataset** adalah dataset klasik dalam Bayesian Network research:

- **Source**: Sachs et al., "Causal Protein-Signaling Networks Derived from Multiparameter Single-Cell Data", _Science_ (2005)
- **Domain**: Flow cytometry measurements dari protein phosphorylation
- **Nodes**: 11 protein/phospholipid molecules
  - PKC, Plcg, PIP2, PIP3, Raf, Mek, Erk, Akt, PKA, Jnk, P38
- **States**: 3 levels per node (LOW, AVG, HIGH)
- **Edges**: 17 directed causal relationships
- **Biological Context**: T-cell signaling pathway

**Struktur Jaringan:**

```
       PKC ──→ PKA ──→ Raf ──→ Mek ──→ Erk ──→ Akt
        │       │       │       │
        ↓       ↓       ↓       ↓
       Jnk     P38     ...    (dll)

      Plcg ──→ PIP3 ──→ PIP2
```

---

# Instalasi

[`^ Kembali ke atas ^`](#top)

## Kebutuhan Sistem

### Minimum Requirements:

- **OS**: Windows 10+, macOS 10.14+, atau Linux (Ubuntu 18.04+)
- **Python**: 3.8 atau lebih baru
- **RAM**: 4GB minimum (8GB+ direkomendasikan)
- **Storage**: 500MB untuk library + dataset
- **Internet**: Koneksi stabil untuk download dependencies

### Prerequisites:

- Python dan pip terinstall
- Git (opsional, untuk clone repository)
- Jupyter Notebook atau VS Code dengan Python extension
- Browser modern (Chrome, Firefox, Safari, Edge)

## Proses Instalasi

### 1. Persiapan Environment

#### Windows:

```bash
# Buka Command Prompt atau PowerShell
# Navigate ke folder yang diinginkan
cd D:\Projects

# Buat virtual environment
python -m venv bayesian_env

# Aktivasi virtual environment
bayesian_env\Scripts\activate
```

#### macOS / Linux:

```bash
# Buka Terminal
# Navigate ke folder yang diinginkan
cd ~/Projects

# Buat virtual environment
python3 -m venv bayesian_env

# Aktivasi virtual environment
source bayesian_env/bin/activate
```

### 2. Clone Repository (Opsional)

Jika menggunakan GitHub:

```bash
git clone https://github.com/naufalarizq/BayesianNetwork-Sachs.git
cd BayesianNetwork-Sachs
```

Atau download ZIP langsung dari GitHub dan ekstrak.

### 3. Install Dependencies

File `requirements.txt` sudah tersedia dengan konten:

```
numpy
pandas
matplotlib
seaborn
pgmpy
networkx
scikit-learn
```

Install semua dependencies:

```bash
pip install -r requirements.txt
```

Atau install secara individual:

```bash
pip install numpy pandas matplotlib seaborn pgmpy networkx scikit-learn jupyter
```

### 4. Setup Dataset

Dataset Sachs sudah tersedia di folder `data/sachs.bif/`:

**Struktur folder:**

```
BayesianNetwork-Sachs/
├── data/
│   ├── generator.py
│   └── sachs.bif/
│       ├── sachs.bif          # BIF file (Bayesian Network structure)
│       ├── sachs.csv          # Generated dataset (5500 samples)
│       ├── sachs_cpts_gen.csv # Learned CPTs dari generated data
│       └── sachs_cpts_ori.csv # Original CPTs (jika ada)
├── Bayesian.py                # Core implementation
├── notebook.ipynb             # Main analysis notebook
├── requirements.txt
└── README.md
```

**Generate dataset (jika belum ada sachs.csv):**

```python
from data.generator import sample

# Generate 5500 samples dari BIF file
sample(r"data\sachs.bif\sachs.bif", 5500, r"data\sachs.bif\sachs.csv")
```

### 5. Jalankan Jupyter Notebook

```bash
# Pastikan virtual environment masih aktif
jupyter notebook

# Browser akan otomatis terbuka ke localhost:8888
# Buka file: notebook.ipynb
```

Atau untuk VS Code:

```bash
# Buka folder di VS Code
code .

# Install Python extension jika belum ada
# Pilih interpreter: bayesian_env
# Buka notebook.ipynb
```

### 6. Verifikasi Instalasi

Jalankan cell pertama dari notebook untuk memverifikasi:

```python
# Cell 1: imports
import re
import numpy as np
import pandas as pd
from collections import OrderedDict
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

print("✓ Semua library berhasil diimport!")
```

Jika tidak ada error, instalasi selesai.

---

# Konfigurasi

[`^ Kembali ke atas ^`](#top)

## Konfigurasi Dataset

### 1. Path Configuration

Pastikan path dataset benar di notebook Cell #3:

```python
# Cell 2: load data
DATA_PATH = r"data\sachs.bif\sachs.csv"
df = pd.read_csv(DATA_PATH)
```

Jika menggunakan path absolut:

```python
import os
dataset_path = os.path.join(os.getcwd(), "data", "sachs.bif", "sachs.csv")
df = pd.read_csv(dataset_path)
```

### 2. Data Generation Configuration

Untuk generate dataset baru dengan ukuran berbeda:

```python
from data.generator import sample

# Parameter:
# - path: BIF file location
# - size: jumlah samples (default: 5500)
# - output_path: CSV output location
sample(r"data\sachs.bif\sachs.bif",
       size=10000,  # Ubah jumlah samples
       output_path=r"data\sachs.bif\sachs_large.csv")
```

### 3. Network Structure Configuration

Parents map (struktur DAG) didefinisikan di Cell #10:

```python
# Cell 4: parents map
parents_map = {
    'PKC': [],              # Root node (no parents)
    'Plcg': [],             # Root node
    'PKA': ['PKC'],         # PKA depends on PKC
    'PIP3': ['Plcg'],
    'Raf': ['PKA', 'PKC'],  # Multiple parents
    'Jnk': ['PKA', 'PKC'],
    'P38': ['PKA', 'PKC'],
    'Mek': ['PKA', 'PKC', 'Raf'],
    'Erk': ['Mek', 'PKA'],
    'PIP2': ['PIP3', 'Plcg'],
    'Akt': ['Erk', 'PKA'],
}
```

**Untuk mengubah struktur jaringan:**

1. Edit parents_map sesuai domain knowledge
2. Pastikan tidak ada cycles (DAG requirement)
3. Re-run parameter learning untuk CPT baru

### 4. CPT Learning Configuration

**Laplace Smoothing Parameter:**

```python
# Cell 5: belajar CPTs
cpts_map, node_states = learn_cpts_from_data(
    df,
    node_list,
    states,
    parents_map,
    laplace=1.0  # Ubah nilai smoothing (0.0 = no smoothing, >1 = strong smoothing)
)
```

**Interpretasi Laplace:**

- `laplace=0.0`: Pure MLE (risk: zero probabilities)
- `laplace=1.0`: Standard Laplace (recommended)
- `laplace=5.0`: Strong smoothing (more uniform distribution)

### 5. State Configuration

Jika dataset memiliki state berbeda dari LOW/AVG/HIGH:

```python
# Cell 4: states configuration
# Default:
states = ['LOW', 'AVG', 'HIGH']

# Untuk binary states:
states = ['0', '1']

# Untuk custom states:
states = ['Inactive', 'Moderate', 'Active', 'VeryActive']
```

### 6. Inference Configuration

Evidence untuk posterior inference (Cell #16):

```python
# Cell 10: inference & plot posterior per node given evidence
evidence = {
    'PKC': 'HIGH',   # Observed state
    'PKA': 'AVG',    # Observed state
    'Raf': 'HIGH'    # Observed state
}

# Untuk multiple evidence scenarios:
evidence_scenarios = [
    {'PKC':'HIGH', 'PKA':'AVG', 'Raf':'HIGH'},
    {'PKC':'LOW', 'PKA':'AVG', 'Raf':'LOW'},
    {'PKC':'AVG', 'PKA':'HIGH', 'Raf':'HIGH'},
]
```

### 7. Visualization Configuration

**Figure sizes:**

```python
# Network visualization (Cell #7):
visualize_bayesian_network(
    cpts_loaded,
    node_states,
    show_cpt_full=False,  # Set True untuk full CPT table
    figsize=(14, 10)      # Ubah size (width, height)
)

# Bar plot sizes (Cell #16):
plt.figure(figsize=(4, 3))  # Ubah untuk chart lebih besar/kecil
```

**Chart resolution:**

```python
# Default:
plt.savefig("output/chart.png", dpi=300, bbox_inches='tight')

# High resolution:
plt.savefig("output/chart.png", dpi=600, bbox_inches='tight')

# Draft/fast:
plt.savefig("output/chart.png", dpi=150, bbox_inches='tight')
```

### 8. Cross-Validation Configuration

K-Fold CV parameters (Cell #15):

```python
# Cell 9: k-fold CV
kf = KFold(
    n_splits=5,         # Ubah jumlah folds (default: 5)
    shuffle=True,       # Set False untuk sequential split
    random_state=42     # Ubah untuk different split
)
```

### 9. Output Configuration

**CSV output paths:**

```python
# Cell 5: save learned CPTs
save_cpts_to_csv(cpts_map, r"data\sachs.bif\sachs_cpts_gen.csv")

# Untuk custom path:
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
save_cpts_to_csv(cpts_map, f"{output_dir}/cpts_custom.csv")
```

---

# Cara Pemakaian

[`^ Kembali ke atas ^`](#top)

## Workflow Tahap Demi Tahap

Lihat file README lengkap untuk panduan detail penggunaan step-by-step meliputi:

- Data Preparation & Generation
- Exploratory Data Analysis
- Parameter Learning
- Network Visualization
- Model Evaluation
- Probabilistic Inference

---

# Pembahasan

[`^ Kembali ke atas ^`](#top)

Proyek ini mengimplementasikan Bayesian Network from scratch menggunakan Python dengan fokus pada:

1. **Parameter Learning**: MLE dengan Laplace Smoothing
2. **Exact Inference**: Enumeration-based posterior computation
3. **Model Evaluation**: Log-Likelihood, BIC, Accuracy, Cross-Validation
4. **Visualization**: NetworkX untuk graph structure dan CPT tables

Untuk pembahasan mendalam tentang metodologi, algoritma, dan hasil analisis, lihat dokumentasi lengkap di repository.

---

# Referensi

[`^ Kembali ke atas ^`](#top)

## Dataset & Sumber Data

1. **Sachs et al. (2005)** - Original Paper

   - Sachs, K., Perez, O., Pe'er, D., Lauffenburger, D. A., & Nolan, G. P. (2005). Causal protein-signaling networks derived from multiparameter single-cell data. _Science_, 308(5721), 523-529.

2. **bnlearn Repository**
   - URL: https://www.bnlearn.com/bnrepository/
   - BIF file format untuk Bayesian Network

## Python Libraries

- [Pandas](https://pandas.pydata.org/docs/) - Data manipulation
- [NumPy](https://numpy.org/doc/) - Numerical computing
- [Matplotlib](https://matplotlib.org/) - Visualization
- [Seaborn](https://seaborn.pydata.org/) - Statistical plots
- [NetworkX](https://networkx.org/) - Graph analysis
- [pgmpy](https://pgmpy.org/) - Probabilistic graphical models
- [Scikit-learn](https://scikit-learn.org/) - Machine learning utilities

## Project Information

**Author:** Naufal Akmal Rizqulloh  
**Institution:** IPB University - Computer Science  
**Course:** Knowledge-Based Systems - Semester 5  
**Repository:** [github.com/naufalarizq/BayesianNetwork-Sachs](https://github.com/naufalarizq/BayesianNetwork-Sachs)  
**Created:** December 2024

---

<p align="center">
  <strong>🧠 Terima kasih telah menggunakan Bayesian Network - Sachs Analysis! 🧬</strong><br>
  Untuk pertanyaan atau feedback, silakan buat issue di repository.
</p>

[`^ Kembali ke atas ^`](#top)
