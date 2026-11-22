**0. Representasi yang dikeluarkan.** Pada tahap 1, mikroskop dapat ditebak dari simclr_gambar, fusi_simclr_concat (akurasi seimbang di atas 0,60). Representasi ini tetap dilaporkan, tetapi tidak dipakai untuk memilih k bersama maupun representasi terbaik, karena klaster di atasnya cenderung memisahkan alat.

**1. Jumlah klaster.** Konsensus empat kriteria (silhouette, Davies-Bouldin, BIC GMM, kestabilan) per representasi: pca_gambar: 4, pca_spektrum: 6, pca_fusi: 4, ae_gambar: 6, simclr_gambar: 8, ae_spektrum: 8, fusi_ae_concat: 2, fusi_simclr_concat: 10, fusi_gabungan: 8. Modus yang dipakai untuk perbandingan: k = 4. Pada pca_gambar, pca_spektrum, pca_fusi, ae_gambar, simclr_gambar, ae_spektrum, fusi_ae_concat, fusi_simclr_concat, fusi_gabungan, setidaknya satu kriteria menunjuk batas rentang (2 atau 12), jadi kriteria itu tidak menemukan puncak yang jelas.

Konsensus tidak seragam antar representasi (k bernilai 2, 4, 6, 8, 10); jumlah klaster perlu dibaca sebagai kisaran, bukan satu angka pasti.

**2. Representasi.** Pada k = 4, representasi dengan peringkat gabungan terbaik adalah pca_spektrum (silhouette 0,17, Davies-Bouldin 1,78, kestabilan 0,83, NMI terhadap mikroskop 0,000). Tiga teratas: pca_spektrum, pca_gambar, simclr_gambar.

Pembanding PCA terbaik berada di peringkat 1 dari 9; PCA tidak kalah dari representasi yang dipelajari menurut ringkasan ini.

**3. Pengaruh mikroskop.** NMI klaster terhadap mikroskop tidak lebih dari 0,05 pada semua representasi, sehingga klaster tidak tampak mengikuti alat. Normalisasi per mikroskop dipakai pada tahap 1.

**3b. Kesepakatan antar modalitas.** pca_gambar vs pca_spektrum: ARI tertinggi 0,000 pada k = 3 (dasar acak -0,000); ae_gambar vs ae_spektrum: ARI tertinggi 0,133 pada k = 6 (dasar acak 0,000); simclr_gambar vs ae_spektrum: ARI tertinggi 0,237 pada k = 8 (dasar acak 0,000). Ada kesepakatan antar modalitas tetapi lemah; kedua modalitas hanya sebagian memuat kelompok yang sama.

**4. Metode.** Pada pca_spektrum dengan k = 4, kesesuaian antar metode (ARI) berkisar dari 0,15 (K-Means dan GMM) sampai 0,43 (K-Means dan Agglomerative (prototipe)). Kesesuaian yang tidak seragam menunjukkan bahwa batas klaster bergantung pada metode, sehingga pilihan metode perlu dipertimbangkan.

**5. Keterbatasan.** Tidak ada label sehingga seluruh ukuran bersifat internal dan tidak membuktikan bahwa klaster sesuai kelompok material sebenarnya. Silhouette dihitung pada sub-sampel, BIC dari GMM diagonal pada PCA 16 dimensi, dan agglomerative dijalankan pada prototipe. Satu seed dipakai untuk model dan K-Means. Hasil bergantung pada embedding tahap 1 dan asumsi normalisasi mikroskop di sana.

**6. Serah-terima.** Tahap 3 (DEC) memakai representasi pca_spektrum dengan k = 4; pusat dan label K-Means tersimpan di `serah_terima/`.