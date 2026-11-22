import numpy as np, pandas as pd
from rul_data import CAP, metrik, skor_asimetris

BATAS_PRED = np.array([0, 20, 40, 60, 80, 100, 125, 1e9])      # bin berdasarkan nilai prediksi
NAMA_BIN = ['<20', '20-39', '40-59', '60-79', '80-99', '100-124', '>=125']

def layak_uji(df, min_siklus=50):
    """Baris yang mirip titik potong data uji: siklus >= min_siklus dan belum rusak (RUL >= 1)."""
    return ((df['siklus'] >= min_siklus) & (df['rul_mentah'] >= 1)).values

def mirip_uji(y, p, grup, layak, n_ulang=20):
    d = pd.DataFrame({'g': np.asarray(grup)[layak], 'i': np.where(layak)[0]})
    r = []
    for rep in range(n_ulang):
        idx = d.groupby('g').sample(1, random_state=rep)['i'].values
        r.append(metrik(y[idx], p[idx]))
    return {k: float(np.mean([x[k] for x in r])) for k in ('rmse', 'skor_total', 'skor_rata', 'bias')}

def _bin(p):
    return np.clip(np.digitize(p, BATAS_PRED) - 1, 0, len(BATAS_PRED) - 2)

# ---------- kalibrasi bias terhadap skor asimetris ----------
def cari_geser(y, p, grid=np.arange(0, 41, 1.0), min_n=200):
    """Untuk tiap bin prediksi, cari geser c (prediksi dikurangi c) yang meminimalkan skor asimetris rata-rata."""
    b = _bin(p)
    out = np.zeros(len(BATAS_PRED) - 1)
    for k in range(len(out)):
        m = b == k
        if m.sum() < min_n:
            continue
        sk = [skor_asimetris(np.clip(p[m] - c, 0, CAP) - y[m]).mean() for c in grid]
        out[k] = grid[int(np.argmin(sk))]
    return out

def terapkan_geser(p, geser):
    p = np.asarray(p, float)
    return np.clip(p - geser[_bin(p)], 0, CAP)

def _separuh(grup, seed=0):
    u = np.unique(grup)
    np.random.default_rng(seed).shuffle(u)
    return np.isin(grup, u[: len(u) // 2])

def skor_silang(y, p, grup, layak):
    """Mesin dibagi dua separuh; geser dicari pada satu separuh dan dinilai pada separuh lain (dua arah)."""
    A = _separuh(grup)
    sebelum, sesudah = [], []
    for fit_m, ev_m in ((A, ~A), (~A, A)):
        g = cari_geser(y[fit_m & layak], p[fit_m & layak])
        ev = ev_m & layak
        sebelum.append(metrik(y[ev], np.clip(p[ev], 0, CAP)))
        sesudah.append(metrik(y[ev], terapkan_geser(p[ev], g)))
    rata = lambda L, k: float(np.mean([x[k] for x in L]))
    return dict(rmse_sebelum=rata(sebelum, 'rmse'), rmse_sesudah=rata(sesudah, 'rmse'),
                skor_sebelum=rata(sebelum, 'skor_rata'), skor_sesudah=rata(sesudah, 'skor_rata'),
                bias_sebelum=rata(sebelum, 'bias'), bias_sesudah=rata(sesudah, 'bias'))

# ---------- interval prediksi dari residu ----------
def tabel_kuantil(y, p, q=(0.1, 0.9), min_n=100):
    b = _bin(p)
    t = np.full((len(BATAS_PRED) - 1, len(q)), np.nan)
    for k in range(t.shape[0]):
        m = b == k
        if m.sum() >= min_n:
            t[k] = np.quantile(y[m] - p[m], q)
    for j in range(t.shape[1]):
        t[:, j] = pd.Series(t[:, j]).interpolate(limit_direction='both').values
    return t

def interval(p, t):
    p = np.asarray(p, float); b = _bin(p)
    return np.clip(p + t[b, 0], 0, CAP), np.clip(p + t[b, 1], 0, CAP)

def liputan_silang(y, p, grup, layak, q=(0.1, 0.9)):
    """Kalibrasi interval pada separuh mesin, ukur liputan dan lebar pada separuh lain (dua arah)."""
    A = _separuh(grup)
    cov, lebar, per_bin = [], [], []
    for fit_m, ev_m in ((A, ~A), (~A, A)):
        t = tabel_kuantil(y[fit_m & layak], p[fit_m & layak], q)
        ev = ev_m & layak
        lo, hi = interval(p[ev], t)
        masuk = (y[ev] >= lo) & (y[ev] <= hi)
        cov.append(masuk.mean()); lebar.append((hi - lo).mean())
        b = _bin(p[ev])
        per_bin.append(pd.DataFrame({'b': b, 'masuk': masuk, 'lebar': hi - lo}).groupby('b').agg(liputan=('masuk', 'mean'), lebar=('lebar', 'mean'), n=('masuk', 'size')))
    pb = sum(x.reindex(range(len(NAMA_BIN))).fillna(0) * w for x, w in zip(per_bin, (0.5, 0.5)))
    pb.index = NAMA_BIN
    return dict(liputan=float(np.mean(cov)), lebar=float(np.mean(lebar))), pb

# ---------- simulasi kebijakan servis ----------
def simulasi_servis(df, p, H, min_siklus=50):
    """Servis pada siklus pertama (>= min_siklus) ketika nilai prediksi p <= H.
    Hasil: sisa umur sebenarnya pada saat servis, per mesin."""
    d = pd.DataFrame({'g': df['id_unit'].values, 's': df['siklus'].values, 'r': df['rul_mentah'].values, 'p': np.asarray(p, float)})
    d = d[d['s'] >= min_siklus]
    sisa = []
    for g, x in d.groupby('g'):
        a = x[x['p'] <= H]
        sisa.append(float(a['r'].iloc[0]) if len(a) else np.nan)
    r = np.array(sisa)
    ada = ~np.isnan(r)
    return dict(H=float(H), tidak_terdeteksi=float(100 * (~ada).mean()),
                terlambat_persen=float(100 * ((r <= 5) | ~ada).mean()),
                sisa_median=float(np.nanmedian(r)) if ada.any() else np.nan,
                sisa_p90=float(np.nanpercentile(r, 90)) if ada.any() else np.nan,
                terbuang_rata=float(np.nanmean(r)) if ada.any() else np.nan)
