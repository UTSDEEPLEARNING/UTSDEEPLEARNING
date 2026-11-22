"""Pengelompokan klasik dan kriteria jumlah klaster untuk Task 3 (tahap 2). Tanpa PyTorch."""
import json, numpy as np, pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, davies_bouldin_score, calinski_harabasz_score,
                             adjusted_rand_score, normalized_mutual_info_score)


def muat_emb(folder):
    man = json.load(open(f'{folder}/manifest.json'))
    emb = {k: np.load(f'{folder}/{k}.npy').astype(np.float32) for k in man['embedding']}
    return emb, man


def kmeans(emb, k, seed=0, n_init=3):
    return KMeans(k, n_init=n_init, random_state=seed).fit(emb)


def gmm_label(emb, k, d_pca=16, seed=0):
    """GMM kovarians diagonal pada PCA d_pca dimensi. Mengembalikan (label, BIC)."""
    z = PCA(min(d_pca, emb.shape[1]), random_state=seed).fit_transform(emb)
    g = GaussianMixture(k, covariance_type='diag', n_init=2, reg_covar=1e-4, random_state=seed).fit(z)
    return g.predict(z), float(g.bic(z))


def aglo_prototipe(emb, k, n_proto=600, seed=0):
    """Agglomerative (Ward) pada n_proto prototipe K-Means, lalu tiap sampel mengikuti prototipenya.
    Ward langsung pada 36.000 sampel memerlukan matriks jarak terlalu besar."""
    km = KMeans(min(n_proto, len(emb) // 5), n_init=1, random_state=seed).fit(emb)
    lab_p = AgglomerativeClustering(k, linkage='ward').fit_predict(km.cluster_centers_)
    return lab_p[km.labels_]


def kestabilan(emb, k, ulang=3, seed=0, frac=0.5):
    """ARI antar dua K-Means yang dilatih pada separuh sampel acak berbeda, keduanya memberi label seluruh data."""
    rng = np.random.RandomState(seed)
    r = []
    for _ in range(ulang):
        p = rng.permutation(len(emb))
        a, b = p[:int(len(emb) * frac)], p[int(len(emb) * frac):]
        ka, kb = KMeans(k, n_init=2, random_state=int(rng.randint(1e6))).fit(emb[a]), KMeans(k, n_init=2, random_state=int(rng.randint(1e6))).fit(emb[b])
        r.append(adjusted_rand_score(ka.predict(emb), kb.predict(emb)))
    return float(np.mean(r))


def kriteria_k(emb, ks, n_sil=8000, seed=0, d_pca=16):
    """Per k: inersia, silhouette (sub-sampel), Davies-Bouldin, Calinski-Harabasz, BIC GMM, kestabilan."""
    rng = np.random.RandomState(seed)
    isub = rng.choice(len(emb), min(n_sil, len(emb)), replace=False)
    out = []
    for k in ks:
        km = kmeans(emb, k, seed)
        _, bic = gmm_label(emb, k, d_pca, seed)
        out.append(dict(k=int(k), inersia=float(km.inertia_),
                        silhouette=float(silhouette_score(emb[isub], km.labels_[isub])),
                        dbi=float(davies_bouldin_score(emb, km.labels_)),
                        ch=float(calinski_harabasz_score(emb, km.labels_)),
                        bic=bic, kestabilan=kestabilan(emb, k, seed=seed)))
    return pd.DataFrame(out)


def k_pilihan(df, ambang_stab=0.8):
    """k yang dipilih tiap kriteria. Kestabilan: k terbesar dengan ARI >= ambang (bila tak ada: k dengan ARI tertinggi).
    Konsensus: median dari empat kriteria, dibulatkan."""
    p = dict(silhouette=int(df.loc[df.silhouette.idxmax(), 'k']),
             dbi=int(df.loc[df.dbi.idxmin(), 'k']),
             bic=int(df.loc[df.bic.idxmin(), 'k']))
    ok = df[df.kestabilan >= ambang_stab]
    p['kestabilan'] = int(ok.k.max()) if len(ok) else int(df.loc[df.kestabilan.idxmax(), 'k'])
    p['konsensus'] = int(round(float(np.median([p['silhouette'], p['dbi'], p['bic'], p['kestabilan']]))))
    return p


def uji_mikroskop(label, mik):
    """ARI dan NMI klaster terhadap mikroskop (dekat 0 = klaster tidak mengikuti alat) serta kemurnian mikroskop terbesar."""
    mik = np.asarray(mik).astype(str)
    lab = np.asarray(label)
    ms = np.unique(mik)
    murni = 0.0
    for c in np.unique(lab):
        i = lab == c
        murni += max((mik[i] == m).sum() for m in ms)
    return dict(ari=float(adjusted_rand_score(mik, lab)), nmi=float(normalized_mutual_info_score(mik, lab)),
                kemurnian=float(murni / len(lab)), mayoritas=float(max((mik == m).mean() for m in ms)))


def ukuran_pada_k(emb, k, mik, seed=0, n_sil=8000):
    km = kmeans(emb, k, seed)
    rng = np.random.RandomState(seed)
    isub = rng.choice(len(emb), min(n_sil, len(emb)), replace=False)
    u = uji_mikroskop(km.labels_, mik)
    return dict(silhouette=float(silhouette_score(emb[isub], km.labels_[isub])), dbi=float(davies_bouldin_score(emb, km.labels_)),
                kestabilan=kestabilan(emb, k, seed=seed), mik_nmi=u['nmi'], mik_ari=u['ari'], mik_kemurnian=u['kemurnian'],
                ukuran_min=float(np.bincount(km.labels_).min() / len(emb))), km


def peringkat_gabungan(df):
    """Rerata peringkat: silhouette tinggi, DBI rendah, kestabilan tinggi, NMI mikroskop rendah."""
    r = pd.DataFrame({'silhouette': df.silhouette.rank(ascending=False), 'dbi': df.dbi.rank(ascending=True),
                      'kestabilan': df.kestabilan.rank(ascending=False), 'mik_nmi': df.mik_nmi.rank(ascending=True)}, index=df.index)
    return r.mean(1)


def kesesuaian_modalitas(ea, eb, ks=range(2, 9), seed=0):
    """Kesepakatan pengelompokan dari dua modalitas: ARI dan NMI antara K-Means pada embedding A dan pada embedding B,
    untuk tiap k. Nilai dasar (permutasi label B) mendekati 0."""
    rng = np.random.RandomState(seed)
    out = []
    for k in ks:
        la, lb = kmeans(ea, k, seed).labels_, kmeans(eb, k, seed).labels_
        out.append(dict(k=int(k), ari=float(adjusted_rand_score(la, lb)), nmi=float(normalized_mutual_info_score(la, lb)),
                        ari_acak=float(adjusted_rand_score(la, rng.permutation(lb)))))
    return pd.DataFrame(out)


def kontribusi_modalitas(lab, refs_g, refs_s):
    """NMI label terhadap klaster satu-modalitas: rata-rata atas referensi gambar dan atas referensi spektrum."""
    g = float(np.mean([normalized_mutual_info_score(lab, r) for r in refs_g]))
    s = float(np.mean([normalized_mutual_info_score(lab, r) for r in refs_s]))
    return g, s


def evaluasi_rep_k(emb, k, mik, refs_g, refs_s, seed=0, n_sil=8000):
    """Mutu K-Means pada satu k, diagnosis mikroskop, dan seberapa seimbang klaster mewakili gambar dan spektrum."""
    u, km = ukuran_pada_k(emb, k, mik, seed, n_sil)
    g, s = kontribusi_modalitas(km.labels_, refs_g, refs_s)
    u.update(nmi_gambar=g, nmi_spektrum=s, seimbang=min(g, s))
    return u, km


def peringkat_k_tetap(df):
    """Rerata peringkat pada k tetap: silhouette tinggi, DBI rendah, kestabilan tinggi, keseimbangan modalitas tinggi."""
    r = pd.DataFrame({'silhouette': df.silhouette.rank(ascending=False), 'dbi': df.dbi.rank(ascending=True),
                      'kestabilan': df.kestabilan.rank(ascending=False), 'seimbang': df.seimbang.rank(ascending=False)}, index=df.index)
    return r.mean(1)
