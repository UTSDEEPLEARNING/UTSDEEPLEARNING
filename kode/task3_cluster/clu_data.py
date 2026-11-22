"""Pemuatan, normalisasi, dan diagnostik tanpa label untuk Task 3 (clustering).
Hanya numpy, pandas, dan scikit-learn; tidak memakai PyTorch."""
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

DATA = '/content/data'


def muat(folder=DATA):
    z = np.load(f'{folder}/clustering.npz')
    g, s = z['gambar'], z['spektrum'].astype(np.float32)
    meta = pd.read_csv(f'{folder}/clustering_meta.csv')
    if 'indeks' in meta.columns:
        meta = meta.sort_values('indeks').reset_index(drop=True)
    assert len(meta) == len(g) == len(s), (len(meta), len(g), len(s))
    return g, s, meta


def isi_spektrum(s):
    """Isi NaN/inf dengan median per fitur. Mengembalikan (spektrum, jumlah sel diisi)."""
    s = np.array(s, dtype=np.float32, copy=True)
    s[~np.isfinite(s)] = np.nan
    n_isi = int(np.isnan(s).sum())
    if n_isi:
        med = np.nanmedian(s, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        idx = np.where(np.isnan(s))
        s[idx] = med[idx[1]]
    return s, n_isi


def siapkan(g, s, mik, per_mikroskop=True):
    """Gambar -> [0,1] lalu z-score skalar; spektrum -> z-score per fitur.
    per_mikroskop=True: statistik dihitung terpisah untuk tiap mikroskop."""
    g = g.astype(np.float32) / 255.0
    s, _ = isi_spektrum(s)
    mik = np.asarray(mik)
    gn, sn = np.empty_like(g), np.empty_like(s)
    grup = [mik == m for m in np.unique(mik)] if per_mikroskop else [np.ones(len(g), bool)]
    for idx in grup:
        gn[idx] = (g[idx] - g[idx].mean()) / (g[idx].std() + 1e-6)
        sn[idx] = (s[idx] - s[idx].mean(0)) / (s[idx].std(0) + 1e-6)
    return gn, sn


def ringkas(g, s, meta):
    mik = meta['mikroskop'].astype(str)
    out = dict(n=int(len(g)), bentuk_gambar=list(g.shape), bentuk_spektrum=list(s.shape),
               dtype_gambar=str(g.dtype), gambar_min=int(g.min()), gambar_maks=int(g.max()),
               spektrum_nan=int((~np.isfinite(s)).sum()),
               id_unik=bool(meta['id_sampel'].is_unique),
               jumlah_mikroskop={k: int(v) for k, v in mik.value_counts().items()})
    st = {}
    for m in sorted(mik.unique()):
        i = (mik == m).values
        gm = g[i].astype(np.float32)
        sp = np.where(np.isfinite(s[i]), s[i], np.nan).astype(np.float64)
        st[m] = dict(gambar_rerata=float(gm.mean()), gambar_simpangan=float(gm.std()),
                     spektrum_rerata=float(np.nanmean(sp)), spektrum_simpangan=float(np.nanstd(sp)))
    out['per_mikroskop'] = st
    return out


def pca_baris(x, d, seed=0):
    x = x.reshape(len(x), -1)
    return PCA(d, svd_solver='randomized', random_state=seed).fit_transform(x).astype(np.float32)


def probe_mikroskop(emb, mik, cv=3, n_maks=15000, seed=0):
    """Seberapa mudah mikroskop ditebak dari representasi (regresi logistik linear, validasi silang)."""
    mik = np.asarray(mik).astype(str)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(emb), min(n_maks, len(emb)), replace=False)
    x, y = StandardScaler().fit_transform(emb[idx]), mik[idx]
    clf = LogisticRegression(max_iter=300, class_weight='balanced')
    p = cross_val_predict(clf, x, y, cv=StratifiedKFold(cv, shuffle=True, random_state=seed))
    _, c = np.unique(y, return_counts=True)
    return dict(akurasi=float(accuracy_score(y, p)), akurasi_seimbang=float(balanced_accuracy_score(y, p)),
                mayoritas=float(c.max() / c.sum()))


def silhouette_k(emb, ks=range(2, 11), n=5000, seed=0):
    rng = np.random.RandomState(seed)
    x = emb[rng.choice(len(emb), min(n, len(emb)), replace=False)]
    return {int(k): float(silhouette_score(x, KMeans(k, n_init=3, random_state=seed).fit_predict(x))) for k in ks}


def tumpang_knn(a, b, k=10, n=6000, seed=0):
    """Rerata irisan k tetangga terdekat antara dua representasi (0..1), beserta nilai kebetulan."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(a), min(n, len(a)), replace=False)
    def tet(x):
        return NearestNeighbors(n_neighbors=k + 1).fit(x[idx]).kneighbors(x[idx], return_distance=False)[:, 1:]
    na, nb = tet(a), tet(b)
    irisan = (na[:, :, None] == nb[:, None, :]).any(2).sum(1) / k
    return dict(tumpang=float(irisan.mean()), kebetulan=float(k / (len(idx) - 1)))


def fusi_concat(eg, es):
    """Gabung dua embedding; tiap blok di-z-score per dimensi lalu dibagi sqrt(dimensi),
    sehingga kedua modalitas punya total varians sama."""
    def blok(e):
        e = (e - e.mean(0)) / (e.std(0) + 1e-6)
        return e / np.sqrt(e.shape[1])
    return np.concatenate([blok(eg), blok(es)], 1).astype(np.float32)


def simpan_emb(folder, nama, arr):
    arr = np.asarray(arr, dtype=np.float32)
    assert np.isfinite(arr).all() and np.abs(arr).max() < 6e4, nama
    np.save(f'{folder}/{nama}.npy', arr.astype(np.float16))
