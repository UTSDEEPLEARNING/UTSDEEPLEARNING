# Kode UTS Deep Learning

Tiga tugas: klasifikasi dua modalitas (Task 1), prediksi sisa umur pakai (Task 2), dan pengelompokan tanpa label (Task 3).
Seluruh angka pada laporan berasal dari notebook di folder ini, dijalankan di Google Colab dengan GPU (T4, L4, atau G4).

## Reproduksi dengan satu perintah

```bash
pip install -r requirements.txt
python kode/reproduksi.py            # semua tugas, berurutan
python kode/reproduksi.py --tugas 2  # hanya satu tugas
python kode/reproduksi.py --daftar   # tampilkan urutan notebook
python kode/reproduksi.py --periksa  # uji cepat pembersihan notebook, tanpa data dan GPU
```

Skrip mengunduh data dari Google Drive (butuh internet), menyalin repositori ke folder kerja (`../uts_dl_kerja`, agar hasil yang
sudah di-commit tidak tertimpa), lalu menjalankan ulang notebook yang sama secara berurutan. Sel khusus Colab dan git
(`google.colab`, `git`, `gdown`, `%%writefile`) dilewati karena berkas `.py` sudah ada di repositori. Keluaran
(`hasil/`, `prediksi_klasifikasi.csv`, `prediksi_rul.csv`, `cluster.csv`) ditulis di folder kerja, dan notebook yang sudah dieksekusi di
`hasil_reproduksi/`. GPU CUDA diperlukan untuk notebook yang melatih jaringan saraf.

Cara lain yang dipakai saat eksperimen: buka tiap notebook di Colab dan jalankan semua sel (Runtime → Run all) menurut urutan di bawah.
Notebook memakai `GH_TOKEN` di Colab Secrets hanya untuk push ke GitHub; tanpa token, sel itu dapat dilewati.

## Urutan notebook dan keluarannya

| Tugas | Notebook (urutan) | Berkas keluaran utama |
|---|---|---|
| 1 Klasifikasi | `task1_fusi` → `task1_domain_noise` → `task1_final` → `task1_sanity` | `prediksi_klasifikasi.csv`, `task1_klasifikasi/bobot/*.pt`, `task1_klasifikasi/hasil/*.json` |
| 2 RUL | `task2_eksplorasi` → `task2_sekuens` | `prediksi_rul.csv`, `task2_rul/hasil/prediksi_rul_interval.csv`, `task2_rul/bobot/*.pt` |
| 3 Clustering | `task3_representasi` → `task3_klaster` → `task3_lanjutan` → `task3_dec` → `task3_verifikasi` | `cluster.csv` (k = 4, salinan dari `task3_cluster/hasil/cluster_k4.csv`), `task3_cluster/emb/*.npy`, `task3_cluster/bobot/*.pt` |

`task1_klasifikasi.ipynb` adalah eksplorasi data awal (contoh citra, nilai hilang, sebaran kelas dan lini). Notebook ini membaca data dari Google Drive pribadi sehingga tidak dijalankan ulang oleh `reproduksi.py`; angka dan gambarnya tersimpan pada output notebook.
Berkas kode dipisah per tugas: `task1_klasifikasi/` (`data.py`, `models.py`, `fusion.py`, `robust.py`, `coteaching.py`, `train_utils.py`),
`task2_rul/` (`rul_data.py`, `rul_eval.py`, `rul_seq.py`), `task3_cluster/` (`clu_data.py`, `clu_model.py`, `clu_cluster.py`, `clu_dec.py`).

## Bobot atau latih ulang

- **Task 1:** bobot ansambel tiga seed di `task1_klasifikasi/bobot/` (fusi gambar-sensor dengan augmentasi). Latih ulang: jalankan `task1_final`.
- **Task 2:** bobot lima fold untuk GRU dan CNN 1D di `task2_rul/bobot/`. Latih ulang: jalankan `task2_sekuens`.
- **Task 3:** bobot encoder (gambar, spektrum, gabungan, SimCLR) di `task3_cluster/bobot/` dan embedding hasilnya di `task3_cluster/emb/`
  (`float16`, urutan baris sama dengan `clustering.npz`). DEC tidak menyimpan bobot; labelnya ditulis ke `cluster.csv` dan dapat dihitung
  ulang dari embedding dengan menjalankan `task3_dec` (sekitar 10 menit di GPU T4).

## Lingkungan, seed, dan determinisme

- Perangkat lunak saat eksperimen: Python 3.13, torch 2.14.1 (CUDA 13.0), numpy 2.4.6, pandas 2.2.3, scikit-learn 1.6.1.
- Seed dicatat di sel konfigurasi tiap notebook (umumnya `SEED = 0`; ansambel memakai seed 0, 1, 2).
- Operasi GPU tidak sepenuhnya deterministik, sehingga angka hasil menjalankan ulang dapat berbeda tipis dari yang dilaporkan.
- Waktu terukur dari log notebook (GPU T4): Task 3 tahap 1 sekitar 31 menit pelatihan encoder; Task 3 tahap 3 sekitar 10 menit. Waktu notebook lain belum dicatat secara terpisah.

## Data

Data dibagikan oleh dosen (`data_uts_dl.zip`) dan tidak disertakan di repositori. `reproduksi.py` mengunduhnya otomatis; atau letakkan hasil unzip di
`data/` dan jalankan dengan `--data data`.
