#!/usr/bin/env python3
"""Reproduksi seluruh angka dengan satu perintah (dijalankan dari akar repositori):

    python kode/reproduksi.py

Notebook di repositori ini awalnya dijalankan di Google Colab (unduh data, git push, sel penulisan berkas kode).
Skrip ini menjalankan ulang notebook yang SAMA secara berurutan, di mesin lokal atau di Colab:
 - berkas kode (*.py) sudah ada di repositori, jadi sel `%%writefile` dilewati;
 - baris khusus Colab/git (google.colab, git, gdown, %cd) diganti `pass`, kode lain pada sel yang sama tetap jalan;
 - jalur /content/data dan /content/UTSDEEPLEARNING dialihkan ke folder data dan salinan kerja;
 - repositori disalin ke folder kerja, sehingga hasil yang sudah di-commit tidak tertimpa.
Notebook yang sudah dieksekusi disimpan di <kerja>/hasil_reproduksi/. Setelah Task 3, berkas
kode/task3_cluster/hasil/cluster_k4.csv (keluaran akhir) disalin menjadi <kerja>/cluster.csv.

Contoh:
  python kode/reproduksi.py                      # semua tugas, berurutan
  python kode/reproduksi.py --tugas 2            # hanya Task 2 (boleh diulang: --tugas 1 --tugas 3)
  python kode/reproduksi.py --daftar             # tampilkan urutan notebook lalu keluar
  python kode/reproduksi.py --periksa            # uji pembersihan notebook tanpa menjalankannya (cepat)
  python kode/reproduksi.py --data /path/data    # pakai data yang sudah diunduh
  python kode/reproduksi.py --lanjut             # lanjut ke notebook berikutnya bila satu gagal

Di Colab (setelah `git clone` dan `%cd` ke akar repositori):  !python kode/reproduksi.py --tugas 2
Butuh GPU (CUDA) untuk notebook yang melatih jaringan saraf; tanpa GPU tetap berjalan tetapi sangat lambat.
Paket: pip install -r requirements.txt  (nbclient, nbformat, dan gdown dipasang otomatis bila belum ada).
"""
import argparse, ast, json, os, shutil, subprocess, sys, time, traceback, zipfile

ID_DRIVE = '10lRGd1QRfLJxwdBh1FaItJih-UgO9vpb'
# Urutan penting: tiap notebook memakai keluaran notebook sebelumnya dalam tugas yang sama.
URUTAN = [
    (1, 'kode/task1_klasifikasi/task1_fusi.ipynb'),
    (1, 'kode/task1_klasifikasi/task1_domain_noise.ipynb'),
    (1, 'kode/task1_klasifikasi/task1_final.ipynb'),
    (1, 'kode/task1_klasifikasi/task1_sanity.ipynb'),
    (2, 'kode/task2_rul/task2_eksplorasi.ipynb'),
    (2, 'kode/task2_rul/task2_sekuens.ipynb'),
    (3, 'kode/task3_cluster/task3_representasi.ipynb'),
    (3, 'kode/task3_cluster/task3_klaster.ipynb'),
    (3, 'kode/task3_cluster/task3_lanjutan.ipynb'),
    (3, 'kode/task3_cluster/task3_dec.ipynb'),
    (3, 'kode/task3_cluster/task3_verifikasi.ipynb'),
]
LEWATI = ('google.colab', 'userdata', 'gdown', '%cd', '!git', 'git clone', 'drive.mount')  # baris khusus Colab/git
BERKAS_DATA = ('klasifikasi_train.npz', 'rul_train.csv.gz', 'clustering.npz')
FOLDER_KELUARAN = ('kode/task1_klasifikasi/hasil', 'kode/task1_klasifikasi/bobot', 'kode/task2_rul/hasil',
                   'kode/task2_rul/bobot', 'kode/task3_cluster/hasil', 'kode/task3_cluster/bobot',
                   'kode/task3_cluster/emb')


def akar_repositori():
    """Cari folder yang berisi kode/ : dari lokasi skrip bila ada, kalau tidak dari folder kerja saat ini."""
    awal = []
    try:
        awal.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:  # dijalankan dari sel notebook
        pass
    awal.append(os.getcwd())
    for d in awal:
        while True:
            if os.path.isdir(os.path.join(d, 'kode', 'task1_klasifikasi')):
                return d
            induk = os.path.dirname(d)
            if induk == d:
                break
            d = induk
    sys.exit('Akar repositori tidak ditemukan. Jalankan dari dalam folder hasil `git clone` repositori ini.')


def sumber(c):
    s = c.get('source', '')
    return ''.join(s) if isinstance(s, list) else s


def bersihkan(nb, repo, data):
    """Siapkan notebook (dict JSON) untuk dijalankan di luar Colab.
    - sel `%%writefile` dibuang (berkas .py sudah ada di repositori);
    - pada sel lain hanya BARIS khusus Colab/git yang diganti `pass`, sisanya (misalnya penulisan hasil) tetap dijalankan;
    - jalur /content dialihkan. Mengembalikan (jumlah sel kode, jumlah baris khusus Colab/git yang diganti)."""
    baru, diganti = [], 0
    for c in nb['cells']:
        if c.get('cell_type') != 'code':
            baru.append(c); continue
        src = sumber(c)
        if src.lstrip().startswith('%%writefile'):
            diganti += 1; continue
        baris = []
        for l in src.splitlines():
            if any(k in l for k in LEWATI):
                baris.append(l[:len(l) - len(l.lstrip())] + 'pass'); diganti += 1
            else:
                baris.append(l)
        src = '\n'.join(baris).replace('/content/UTSDEEPLEARNING', repo).replace('/content/data', data)
        c['source'], c['outputs'], c['execution_count'] = src, [], None
        baru.append(c)
    nb['cells'] = baru
    return sum(c.get('cell_type') == 'code' for c in baru), diganti


def sintaks_ok(nb):
    """Periksa sintaks tiap sel kode (baris sihir ! dan % diabaikan). Mengembalikan daftar sel bermasalah."""
    rusak = []
    for i, c in enumerate(nb['cells']):
        if c.get('cell_type') != 'code': continue
        baris = [('' if l.lstrip().startswith(('!', '%')) else l) for l in sumber(c).splitlines()]
        try:
            ast.parse('\n'.join(baris))
        except SyntaxError as e:
            rusak.append((i, str(e)))
    return rusak


def pastikan_paket():
    for modul, paket in (('nbformat', 'nbformat'), ('nbclient', 'nbclient'), ('ipykernel', 'ipykernel')):
        try:
            __import__(modul)
        except ImportError:
            print(f'Memasang {paket} ...', flush=True)
            subprocess.check_call([sys.executable, '-m', 'pip', '-q', 'install', paket])


def unduh_data(data):
    if all(os.path.exists(os.path.join(data, f)) for f in BERKAS_DATA):
        return
    try:
        import gdown
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', '-q', 'install', 'gdown'])
        import gdown
    os.makedirs(data, exist_ok=True)
    z = os.path.join(data, 'data_uts_dl.zip')
    print('Mengunduh data dari Google Drive ...', flush=True)
    gdown.download(id=ID_DRIVE, output=z, quiet=False)
    with zipfile.ZipFile(z) as f:
        f.extractall(data)
    # beberapa arsip memuat satu folder induk; ratakan bila berkas data tidak langsung di dalam `data`
    if not all(os.path.exists(os.path.join(data, f)) for f in BERKAS_DATA):
        for akar, _, files in os.walk(data):
            for f in files:
                if f in BERKAS_DATA and akar != data:
                    shutil.copy(os.path.join(akar, f), os.path.join(data, f))
    hilang = [f for f in BERKAS_DATA if not os.path.exists(os.path.join(data, f))]
    if hilang:
        sys.exit(f'Berkas data tidak ditemukan setelah unduh: {hilang}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tugas', type=int, choices=[1, 2, 3], action='append', help='ulangi untuk beberapa tugas; default semua')
    ap.add_argument('--data', default=None, help='folder data (default: <repo>/data)')
    ap.add_argument('--kerja', default=None, help='folder kerja (default: <repo>/../uts_dl_kerja)')
    ap.add_argument('--daftar', action='store_true', help='tampilkan urutan notebook lalu keluar')
    ap.add_argument('--periksa', action='store_true', help='uji pembersihan dan sintaks notebook tanpa menjalankan')
    ap.add_argument('--lanjut', action='store_true', help='lanjut ke notebook berikutnya bila satu gagal')
    ap.add_argument('--timeout', type=int, default=-1, help='batas waktu per sel (detik); -1 = tanpa batas')
    a, _ = ap.parse_known_args()  # abaikan argumen bawaan kernel Jupyter/Colab (-f ...)

    repo_asal = akar_repositori()
    daftar = [(t, p) for t, p in URUTAN if a.tugas is None or t in a.tugas]
    if a.daftar:
        for t, p in daftar: print(f'Task {t}: {p}')
        return
    data = os.path.abspath(a.data or os.path.join(repo_asal, 'data'))
    kerja = os.path.abspath(a.kerja or os.path.join(repo_asal, '..', 'uts_dl_kerja'))

    if a.periksa:
        masalah = 0
        for t, p in daftar:
            nb = json.load(open(os.path.join(repo_asal, p), encoding='utf-8'))
            n, lewat = bersihkan(nb, kerja, data)
            sisa = [i for i, c in enumerate(nb['cells']) if c.get('cell_type') == 'code' and '/content' in sumber(c)]
            rusak = sintaks_ok(nb)
            masalah += len(sisa) + len(rusak)
            print(f'Task {t}: {p}: {n} sel dijalankan, {lewat} baris/sel khusus Colab diganti, jalur /content tersisa {sisa}, sintaks rusak {rusak}')
        print('PERIKSA', 'LULUS' if masalah == 0 else f'ADA {masalah} MASALAH'); return

    pastikan_paket()
    import nbformat
    from nbclient import NotebookClient
    unduh_data(data)
    if os.path.exists(kerja): shutil.rmtree(kerja)
    shutil.copytree(repo_asal, kerja, ignore=shutil.ignore_patterns('.git', 'data', '__pycache__', 'hasil_reproduksi'))
    for d in FOLDER_KELUARAN: os.makedirs(os.path.join(kerja, d), exist_ok=True)  # pengganti `!mkdir` pada sel yang dilewati
    out = os.path.join(kerja, 'hasil_reproduksi'); os.makedirs(out, exist_ok=True)
    print('Salinan kerja:', kerja, '| data:', data, flush=True)

    gagal = []
    for t, p in daftar:
        nb = json.load(open(os.path.join(kerja, p), encoding='utf-8'))
        n, lewat = bersihkan(nb, kerja, data)
        print(f'\n=== Task {t}: {p} ({n} sel dijalankan, {lewat} baris/sel khusus Colab/git diganti) ===', flush=True)
        nbk = nbformat.from_dict(nb)
        t0 = time.time()
        try:
            NotebookClient(nbk, timeout=None if a.timeout < 0 else a.timeout, kernel_name='python3',
                           resources={'metadata': {'path': kerja}}).execute()
            print(f'selesai dalam {time.time() - t0:.0f} dtk', flush=True)
        except Exception:
            gagal.append(p)
            print(f'GAGAL pada {p}:\n{traceback.format_exc()}', flush=True)
        nbformat.write(nbk, os.path.join(out, os.path.basename(p)))  # simpan juga yang gagal, untuk diperiksa
        if p in gagal and not a.lanjut:
            break
        if t == 3 and p.endswith('task3_verifikasi.ipynb') and p not in gagal:
            k4 = os.path.join(kerja, 'kode/task3_cluster/hasil/cluster_k4.csv')
            if os.path.exists(k4):
                shutil.copy(k4, os.path.join(kerja, 'cluster.csv'))
                print('cluster.csv (k = 4) disalin dari hasil/cluster_k4.csv', flush=True)

    print('\nKeluaran ada di', kerja, '(hasil/, prediksi_*.csv, cluster.csv); notebook yang dieksekusi di', out)
    if gagal:
        print('NOTEBOOK GAGAL:', gagal); sys.exit(1)
    print('Semua notebook selesai.')


if __name__ == '__main__':
    main()
