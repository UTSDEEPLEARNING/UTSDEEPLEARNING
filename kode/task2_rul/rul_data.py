import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SENS = [f's{i:02d}' for i in range(1, 22)]
SET = ['setel_1', 'setel_2', 'setel_3']
CAP = 150
BINS = [(0, 30), (30, 60), (60, 100), (100, 150), (150, 10**6)]       # [a, b)
BIN_NAMES = ['0-29', '30-59', '60-99', '100-149', '150 (batas atas)']

def load(data_dir='/content/data'):
    tr = pd.read_csv(f'{data_dir}/rul_train.csv.gz')
    te = pd.read_csv(f'{data_dir}/rul_test.csv.gz')
    tr = tr.sort_values(['id_unit', 'siklus']).reset_index(drop=True)
    te = te.sort_values(['id_unit', 'siklus']).reset_index(drop=True)
    return tr, te

def add_rul(df, cap=CAP):
    """Label RUL data latih: siklus terakhir tiap mesin = siklus rusak (RUL 0), dipotong pada cap."""
    T = df.groupby('id_unit')['siklus'].transform('max')
    df = df.copy()
    df['rul_mentah'] = T - df['siklus']
    df['rul'] = df['rul_mentah'].clip(upper=cap)
    return df

# ---------- metrik ----------
def skor_asimetris(d):
    d = np.asarray(d, float)
    return np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)      # d = prediksi - aktual

def metrik(y, p):
    d = np.asarray(p, float) - np.asarray(y, float)
    s = skor_asimetris(d)
    return dict(rmse=float(np.sqrt(np.mean(d ** 2))), skor_total=float(s.sum()),
                skor_rata=float(s.mean()), bias=float(d.mean()), n=int(len(d)))

def metrik_per_rentang(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    rows = []
    for (a, b), nm in zip(BINS, BIN_NAMES):
        m = (y >= a) & (y < b)
        if m.sum() == 0:
            continue
        r = metrik(y[m], p[m]); r['rentang'] = nm
        rows.append(r)
    return pd.DataFrame(rows).set_index('rentang')

# ---------- normalisasi bergantung kondisi operasi ----------
def _mad(v):
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return 1.0
    s = 1.4826 * np.median(np.abs(v - np.median(v)))
    return s if s > 1e-12 else 1.0

def pilih_k_regime(df, kmax=8, n_sampel=6000, seed=0):
    Z = StandardScaler().fit_transform(df[SET].values)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), size=min(n_sampel, len(Z)), replace=False)
    hasil = {}
    for k in range(2, kmax + 1):
        lab = KMeans(k, n_init=5, random_state=seed).fit_predict(Z[idx])
        hasil[k] = float(silhouette_score(Z[idx], lab))
    return hasil

class Normalisasi:
    """mode='regime': z-score robust per regime operasi (klaster pada setel_1..3).
       mode='regresi': residu regresi linear sensor terhadap setel_1..3, lalu z-score."""
    def __init__(self, mode, k=None, seed=0):
        self.mode, self.k, self.seed = mode, k, seed

    def fit(self, df, cols):
        self.cols = list(cols)
        if self.mode == 'regime':
            self.sc = StandardScaler().fit(df[SET].values)
            self.km = KMeans(self.k, n_init=10, random_state=self.seed).fit(self.sc.transform(df[SET].values))
            r = self.km.predict(self.sc.transform(df[SET].values))
            g = df[self.cols].groupby(r)
            self.mu = g.median()
            self.sd = g.agg(lambda s: _mad(s.values))
        else:
            self.lr = {}
            for c in self.cols:
                m = df[c].notna().values
                X = np.c_[np.ones(m.sum()), df.loc[m, SET].values]
                coef = np.linalg.lstsq(X, df.loc[m, c].values, rcond=None)[0]
                self.lr[c] = (coef, _mad(df.loc[m, c].values - X @ coef))
        return self

    def regime(self, df):
        return self.km.predict(self.sc.transform(df[SET].values))

    def transform(self, df):
        out = df.copy()
        if self.mode == 'regime':
            r = self.regime(df)
            out[self.cols] = (df[self.cols].values - self.mu.loc[r].values) / self.sd.loc[r].values
            out['regime'] = r
        else:
            X = np.c_[np.ones(len(df)), df[SET].values]
            for c in self.cols:
                coef, s = self.lr[c]
                out[c] = (df[c].values - X @ coef) / s
        return out

# ---------- pembersihan kausal ----------
def despike(df, cols, w=5, k=6.0, scale=None):
    """Lonjakan = selisih terhadap median w siklus SEBELUMNYA > k x skala robust. Hanya memakai masa lalu."""
    prev_med = df.groupby('id_unit')[cols].transform(lambda s: s.shift(1).rolling(w, min_periods=1).median())
    res = df[cols] - prev_med
    if scale is None:
        scale = 1.4826 * (res - res.median()).abs().median()
        scale = scale.where(scale > 1e-12, 1.0)
    flag = res.abs() > k * scale
    out = df.copy()
    out[cols] = df[cols].where(~flag, prev_med)
    return out, flag.sum(), scale

def isi_kausal(df, cols):
    """Isi nilai hilang dengan nilai terakhir yang ada pada mesin yang sama; sisa (awal riwayat) = 0 (rata-rata regime)."""
    out = df.copy()
    out[cols] = out.groupby('id_unit')[cols].ffill().fillna(0.0)
    return out

def tren_per_sensor(df, cols):
    """Median korelasi Spearman nilai sensor vs nomor siklus per mesin (tanpa label, tanpa data uji)."""
    r = df.groupby('id_unit').apply(lambda g: g[cols].corrwith(g['siklus'], method='spearman'))
    return r.median()

# ---------- fitur jendela (baseline non-DL) ----------
def fitur_jendela(df, cols, W=30):
    g = df.groupby('id_unit')
    X = pd.DataFrame(index=df.index)
    X['siklus'] = df['siklus'].values
    for c in SET:
        X[c] = df[c].values
    for c in cols:
        s = g[c]
        X[c + '_last'] = df[c].values
        X[c + '_mean'] = s.transform(lambda v: v.rolling(W, min_periods=1).mean()).values
        X[c + '_std'] = s.transform(lambda v: v.rolling(W, min_periods=2).std()).fillna(0.0).values
        awal = s.shift(W - 1).fillna(s.transform('first'))
        X[c + '_diff'] = (df[c] - awal).values
    return X

# ---------- jendela geser untuk model sekuens ----------
def buat_jendela(df, cols, W=30):
    """Jendela W siklus yang berakhir di tiap siklus. Awal riwayat diisi dengan baris pertama mesin itu
    (masa lalu, bukan masa depan) dan ditandai kanal mask (0 = isian). Mengembalikan X (N, W, F+1), y, id, siklus."""
    Xs, ys, us, ts = [], [], [], []
    for u, g in df.groupby('id_unit', sort=False):
        a = g[cols].values.astype(np.float32)
        n = len(a)
        pad = np.vstack([np.repeat(a[:1], W - 1, axis=0), a]) if W > 1 else a
        mask = np.r_[np.zeros(W - 1, np.float32), np.ones(n, np.float32)]
        idx = np.arange(n)[:, None] + np.arange(W)[None, :]
        Xs.append(np.concatenate([pad[idx], mask[idx][..., None]], axis=-1))
        ys.append(g['rul'].values.astype(np.float32) if 'rul' in g.columns else np.full(n, np.nan, np.float32))
        us.append(np.full(n, u)); ts.append(g['siklus'].values)
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(us), np.concatenate(ts)
