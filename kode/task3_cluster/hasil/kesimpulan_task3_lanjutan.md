**1. Jumlah klaster.** Kesepakatan klaster antara gambar dan spektrum (rata-rata ARI empat pasangan) tertinggi pada k* = 9 (ARI 0,168, dasar acak mendekati 0). Puncaknya tegas dan tidak berada di batas rentang. Himpunan k calon untuk DEC: 7, 9, 11. BIC terendah jatuh di batas atas rentang pada 89% representasi, sehingga tidak dipakai sebagai kriteria.

**k = 7.** Tiga teratas yang lolos gerbang mikroskop: fusi_simclr_concat (silhouette 0,16, kestabilan 0,85, keseimbangan 0,411), fusi_gabungan (silhouette 0,10, kestabilan 0,85, keseimbangan 0,343), pca_spektrum (silhouette 0,18, kestabilan 0,52, keseimbangan 0,212).

**k = 9.** Tiga teratas yang lolos gerbang mikroskop: fusi_simclr_concat (silhouette 0,16, kestabilan 0,90, keseimbangan 0,446), fusi_gabungan (silhouette 0,09, kestabilan 0,89, keseimbangan 0,419), pca_spektrum (silhouette 0,18, kestabilan 0,72, keseimbangan 0,285).

**k = 11.** Tiga teratas yang lolos gerbang mikroskop: fusi_gabungan (silhouette 0,09, kestabilan 0,78, keseimbangan 0,413), pca_spektrum (silhouette 0,18, kestabilan 0,69, keseimbangan 0,296), ae_spektrum (silhouette 0,12, kestabilan 0,69, keseimbangan 0,338).

**2. Gerbang mikroskop.** Representasi yang klasternya mengikuti mikroskop (NMI di atas 0,05) pada setidaknya satu k calon: fusi_simclr_concat, simclr_gambar.

**3. Keterbatasan.** Seluruh ukuran bersifat internal. Keseimbangan modalitas adalah heuristik yang mengandaikan satu kelompok laten memengaruhi kedua modalitas; bila gambar dan spektrum sebenarnya menggambarkan kelompok yang berbeda, ukuran ini tidak memihak klaster yang benar. Kesepakatan antar modalitas dihitung dari K-Means yang bisa berbeda dengan metode yang akhirnya dipakai. Satu seed.

**4. Serah-terima.** Daftar calon (representasi × k) beserta pusat dan label K-Means tersimpan di `serah_terima_v2/`. Tahap 3 menjalankan DEC pada calon tersebut dan memilih konfigurasi akhir.