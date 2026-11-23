# Kontribusi

Kelompok: **Herpratama Rafi Putranto** (`rafiptrnto`) dan **Andrian Irmawan** (`siojzz`). Semua pekerjaan dilakukan di fork masing-masing dan digabung ke `main` lewat pull request (semua PR berstatus *merged*, tidak ada yang terbuka).
Daftar lengkap PR: https://github.com/UTSDEEPLEARNING/UTSDEEPLEARNING/pulls?q=is%3Apr

## Peran dan pull request

### Rafi (`rafiptrnto`)
- **Task 1, klasifikasi (seluruhnya):** eksplorasi, model citra dan sensor, fusi fitur dan keputusan, *modality dropout*, pergeseran domain, label noise (*label smoothing*, *co-teaching*), model akhir tiga seed, `prediksi_klasifikasi.csv`, uji kewajaran satu batch. PR [#3](https://github.com/UTSDEEPLEARNING/UTSDEEPLEARNING/pull/3), [#4](https://github.com/UTSDEEPLEARNING/UTSDEEPLEARNING/pull/4), [#5](https://github.com/UTSDEEPLEARNING/UTSDEEPLEARNING/pull/5), [#6](https://github.com/UTSDEEPLEARNING/UTSDEEPLEARNING/pull/6), [#7](https://github.com/UTSDEEPLEARNING/UTSDEEPLEARNING/pull/7), [#8](https://github.com/UTSDEEPLEARNING/UTSDEEPLEARNING/pull/8), [#9](https://github.com/UTSDEEPLEARNING/UTSDEEPLEARNING/pull/9), [#10](https://github.com/UTSDEEPLEARNING/UTSDEEPLEARNING/pull/10), [#11](https://github.com/UTSDEEPLEARNING/UTSDEEPLEARNING/pull/11), [#12](https://github.com/UTSDEEPLEARNING/UTSDEEPLEARNING/pull/12), [#13](https://github.com/UTSDEEPLEARNING/UTSDEEPLEARNING/pull/13), [#14](https://github.com/UTSDEEPLEARNING/UTSDEEPLEARNING/pull/14), [#25](https://github.com/UTSDEEPLEARNING/UTSDEEPLEARNING/pull/25), [#26](https://github.com/UTSDEEPLEARNING/UTSDEEPLEARNING/pull/26).
- **Task 3, clustering, tahap 2, 2b, dan verifikasi:** perbandingan K-Means, GMM, dan aglomeratif; kriteria jumlah klaster; kesepakatan antar modalitas; diagnosis pengaruh mikroskop; eksplorasi distorsi data, pola spektrum per klaster, dan keputusan akhir k = 4. PR [#18](https://github.com/UTSDEEPLEARNING/UTSDEEPLEARNING/pull/18), [#19](https://github.com/UTSDEEPLEARNING/UTSDEEPLEARNING/pull/19), [#20](https://github.com/UTSDEEPLEARNING/UTSDEEPLEARNING/pull/20), [#21](https://github.com/UTSDEEPLEARNING/UTSDEEPLEARNING/pull/21), [#23](https://github.com/UTSDEEPLEARNING/UTSDEEPLEARNING/pull/23), [#24](https://github.com/UTSDEEPLEARNING/UTSDEEPLEARNING/pull/24).
- **Laporan dan reproduksi:** `laporan.pdf` (PR [#28](https://github.com/UTSDEEPLEARNING/UTSDEEPLEARNING/pull/28)); perbaikan `kode/reproduksi.py` (PR [#29](https://github.com/UTSDEEPLEARNING/UTSDEEPLEARNING/pull/29)); uji reproduksi Task 2 di Google Colab (GPU T4). PR uji alur [#1](https://github.com/UTSDEEPLEARNING/UTSDEEPLEARNING/pull/1).

### Rian (`siojzz`)
- **Task 2, regresi RUL (seluruhnya):** eksplorasi, label RUL, jendela geser tanpa kebocoran, baseline, model GRU dan CNN 1D, kalibrasi skor asimetris, interval prediksi, kebijakan servis, `prediksi_rul.csv`. PR [#15](https://github.com/UTSDEEPLEARNING/UTSDEEPLEARNING/pull/15), [#16](https://github.com/UTSDEEPLEARNING/UTSDEEPLEARNING/pull/16).
- **Task 3, clustering, tahap 1:** representasi tanpa label (*autoencoder*, SimCLR, spektrum) dan fusi modalitas. PR [#17](https://github.com/UTSDEEPLEARNING/UTSDEEPLEARNING/pull/17).
- **Task 3, clustering, tahap 3:** DEC dan pemilihan konfigurasi akhir. PR [#22](https://github.com/UTSDEEPLEARNING/UTSDEEPLEARNING/pull/22).
- **Dokumen repositori dan hasil akhir:** `cluster.csv` akhir (k = 4), `README.md`, `requirements.txt`, `kode/README.md`, dan `kode/reproduksi.py`. PR [#27](https://github.com/UTSDEEPLEARNING/UTSDEEPLEARNING/pull/27). PR uji alur [#2](https://github.com/UTSDEEPLEARNING/UTSDEEPLEARNING/pull/2).

Tahap dalam satu tugas dikerjakan berurutan karena tiap tahap memakai keluaran tahap sebelumnya; di luar itu, pembagian di atas adalah pembagian kerja kami. Keputusan memakai k = 4 pada Task 3 diambil bersama setelah tahap verifikasi dan dicatat di laporan sebagai keputusan post-hoc.

## Penggunaan alat AI

Kami memakai **Claude (Anthropic)** sebagai asisten. Perannya: merancang eksperimen bersama kami, menulis dan menyunting kode notebook dan modul Python, menulis kesimpulan otomatis pada notebook, dan menyusun draf laporan serta dokumen repositori (termasuk berkas ini).
Kami berdua menjalankan semua eksperimen di Google Colab dengan akun masing-masing, memeriksa keluaran dan galat, memutuskan arah eksperimen, dan membuat serta menggabungkan semua pull request. Angka pada laporan diambil dari keluaran notebook yang kami jalankan, bukan dari alat AI.
Pemakaian ini disetujui oleh semua anggota kelompok (Rafi dan Rian).
