import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

KELAS = ['normal', 'gores', 'noda', 'retak', 'gelembung', 'las_dingin', 'kontaminasi']
K2I = {k: i for i, k in enumerate(KELAS)}

def load(split, data_dir="/content/data"):
    z = np.load(f"{data_dir}/klasifikasi_{split}.npz")
    meta = pd.read_csv(f"{data_dir}/klasifikasi_{split}_meta.csv")
    img = z['gambar'].astype(np.float32) / 255.0      # (N, 64, 64)
    sen = z['sensor'].astype(np.float32)               # (N, 3, 128), NaN = tidak terekam
    y = meta['kelas'].map(K2I).values if 'kelas' in meta.columns else None
    return img, sen, meta, y

def sensor_stats(sen):
    return (np.nanmean(sen, axis=(0, 2), keepdims=True),
            np.nanstd(sen, axis=(0, 2), keepdims=True))

def prep_sensor(sen, mean, std):
    mask = (~np.isnan(sen)).astype(np.float32)          # 1 = ada data
    x = np.nan_to_num((sen - mean) / std, nan=0.0)
    return np.concatenate([x, mask], axis=1)            # (N, 6, 128)

def make_split(meta, y, seed=0, frac=0.2):
    strat = meta['kelas'].astype(str) + '_' + meta['lini']
    tr, va = train_test_split(np.arange(len(meta)), test_size=frac,
                              stratify=strat, random_state=seed)
    return tr, va
