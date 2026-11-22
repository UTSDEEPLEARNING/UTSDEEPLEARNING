import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from rul_data import *

def siapkan(tr, te, k=6, thr=0.1):
    """Pipeline pra-pemrosesan yang sama dengan notebook eksplorasi: normalisasi per regime, despike kausal,
    pengisian kausal, seleksi sensor berdasarkan tren (tanpa label). Menambahkan one-hot regime dan umur."""
    nm = Normalisasi('regime', k).fit(tr, SENS)
    trn, ten = nm.transform(tr), nm.transform(te)
    trn, _, skala = despike(trn, SENS)
    ten, _, _ = despike(ten, SENS, scale=skala)
    trc, tec = isi_kausal(trn, SENS), isi_kausal(ten, SENS)
    tren = tren_per_sensor(trc, SENS)
    pakai = [c for c in SENS if abs(tren[c]) >= thr]
    for d in (trc, tec):
        for r in range(k):
            d[f'reg{r}'] = (d['regime'] == r).astype(np.float32)
        d['umur'] = d['siklus'] / 500.0
    kolom = pakai + [f'reg{r}' for r in range(k)] + ['umur']
    return trc, tec, pakai, kolom

class GRURegresi(nn.Module):
    def __init__(self, f, h=64, p=0.2):
        super().__init__()
        self.gru = nn.GRU(f, h, num_layers=2, batch_first=True, dropout=p)
        self.head = nn.Sequential(nn.Linear(h, 64), nn.ReLU(), nn.Dropout(p), nn.Linear(64, 1))
    def forward(self, x):
        o, _ = self.gru(x)
        return self.head(o[:, -1]).squeeze(1)

class CNNRegresi(nn.Module):
    """CNN 1D kausal-longgar dengan dilasi 1, 2, 4, 8 (bidang reseptif 31 siklus)."""
    def __init__(self, f, h=64, p=0.2):
        super().__init__()
        def blk(i, o, d):
            return nn.Sequential(nn.Conv1d(i, o, 3, padding=d, dilation=d), nn.BatchNorm1d(o), nn.ReLU())
        self.body = nn.Sequential(blk(f, h, 1), blk(h, h, 2), blk(h, h, 4), blk(h, h, 8))
        self.head = nn.Sequential(nn.Linear(2 * h, 64), nn.ReLU(), nn.Dropout(p), nn.Linear(64, 1))
    def forward(self, x):
        z = self.body(x.transpose(1, 2))
        return self.head(torch.cat([z[:, :, -1], z.mean(2)], dim=1)).squeeze(1)

@torch.no_grad()
def prediksi(model, X, device='cuda', bs=4096):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        out.append(model(torch.from_numpy(X[i:i + bs]).to(device)).float().cpu())
    return torch.cat(out).numpy() * CAP

def latih_fold(model, Xtr, ytr, Xva, yva, epochs=12, lr=2e-3, bs=1024, device='cuda', seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    model.to(device)
    Xt = torch.from_numpy(Xtr).to(device)
    yt = torch.from_numpy((ytr / CAP).astype(np.float32)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(Xt) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    hist = []
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xt), device=device); tot = 0.0
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            loss = F.mse_loss(model(Xt[idx]), yt[idx])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item() * len(idx)
        pv = np.clip(prediksi(model, Xva, device), 0, CAP)
        rm = float(np.sqrt(np.mean((pv - yva) ** 2)))
        hist.append((tot / len(Xt), rm))
        print(f"  ep {ep + 1}: loss {hist[-1][0]:.5f} | RMSE validasi {rm:.2f}")
    del Xt, yt
    torch.cuda.empty_cache()
    return model, hist

def cv_sekuens(fn, X, y, grup, Xte, nama, slug, dir_bobot, n_fold=5, epochs=12, device='cuda'):
    """Validasi silang per mesin. Mengembalikan prediksi out-of-fold (RUL), prediksi uji tiap fold, dan riwayat latih.
    Bobot tiap fold disimpan; kelima model dipakai sebagai ansambel pada data uji."""
    oof = np.zeros(len(y), dtype=np.float32); te_pred, hist = [], []
    for f, (a, b) in enumerate(GroupKFold(n_fold).split(X, y, grup)):
        print(f"{nama} | fold {f + 1}/{n_fold}")
        model, h = latih_fold(fn(), X[a], y[a], X[b], y[b], epochs=epochs, device=device, seed=f)
        oof[b] = prediksi(model, X[b], device)
        te_pred.append(prediksi(model, Xte, device))
        torch.save(model.state_dict(), f"{dir_bobot}/{slug}_fold{f}.pt")
        hist.append(h)
        del model
        torch.cuda.empty_cache()
    return oof, np.array(te_pred), hist
