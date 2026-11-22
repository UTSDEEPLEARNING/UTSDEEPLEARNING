**1. Jumlah klaster.** Jumlah klaster akhir adalah 9, yaitu k* dari tahap 2b. Himpunan k calon yang dievaluasi: 7, 9, 11.

**2. DEC dibanding K-Means.** Pada 9 kombinasi (representasi × k), silhouette DEC lebih tinggi daripada K-Means pada 3 kombinasi; selisih rata-ratanya -0,028. Silhouette dihitung pada ruang embedding asal, bukan ruang laten DEC, sehingga peningkatan tidak otomatis karena DEC mengoptimalkan ukuran itu.

Pada 6 dari 27 pelaksanaan, ada klaster yang menciut di bawah 2% data (kecenderungan runtuhnya DEC pada struktur lemah); pelaksanaan itu dikeluarkan dari pemilihan.

**3. Konfigurasi akhir.** Representasi fusi_simclr_concat dengan K-Means dan k = 9 (silhouette 0,16, Davies-Bouldin 2,02, keseimbangan modalitas 0,446, NMI terhadap mikroskop 0,049). Kestabilan antar empat seed (ARI berpasangan) 0,96; pelabelan yang diserahkan berasal dari seed 1 (paling mewakili).

**4. Metode pada konfigurasi akhir.** Kesesuaian antar lima pembagian (ARI) berkisar dari 0,44 (K-Means dan Agglomerative (prototipe)) sampai 0,99 (DEC dan DEC+rek). Kesesuaian tidak seragam, sehingga batas klaster bergantung pada metode.

**5. Keluaran.** `cluster.csv` berisi 36000 baris dengan 9 klaster; ukuran klaster: {0: 1261, 1: 3577, 2: 5591, 3: 5098, 4: 4896, 5: 3679, 6: 4988, 7: 2394, 8: 4516}.

**6. Keterbatasan.** Tidak ada label, sehingga seluruh ukuran bersifat internal dan tidak membuktikan kesesuaian dengan kelompok material sebenarnya; skor ARI terhadap label tersembunyi hanya diketahui setelah penilaian. DEC dijalankan di atas embedding tetap dari tahap 1 (encoder gambar dan spektrum tidak dilatih ulang bersama DEC). Jumlah klaster ditetapkan dari kesepakatan antar modalitas, yang mengandaikan satu kelompok laten memengaruhi gambar dan spektrum, dan kepekaan terhadap k hanya diperiksa pada beberapa nilai. Keseimbangan modalitas adalah heuristik. Hiperparameter DEC tidak disetel.