import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score
from fusion import drop_sensor
from robust import get_loader3, predict3, prep_img

def fit_coteaching(m1, m2, Xi, Xs, y, lc, tr_idx, va_idx, epochs=15, lr=1e-3, bs=256,
                   class_weight=None, p_drop=0.3, aug=False, forget=0.10, warm=5,
                   device='cuda', seed=0, y_eval=None):
    torch.manual_seed(seed); np.random.seed(seed)
    y_eval = y if y_eval is None else y_eval
    m1.to(device); m2.to(device)
    wv = torch.ones(7, device=device) if class_weight is None else torch.tensor(class_weight, dtype=torch.float32, device=device)
    o1 = torch.optim.AdamW(m1.parameters(), lr=lr, weight_decay=1e-4)
    o2 = torch.optim.AdamW(m2.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(tr_idx) / bs))
    s1 = torch.optim.lr_scheduler.OneCycleLR(o1, max_lr=lr, total_steps=steps)
    s2 = torch.optim.lr_scheduler.OneCycleLR(o2, max_lr=lr, total_steps=steps)
    tl = get_loader3(Xi, Xs, y, lc, tr_idx, bs, True)
    vl = get_loader3(Xi, Xs, y_eval, lc, va_idx)
    hist = []
    for ep in range(epochs):
        rate = forget * min(1.0, ep / warm)
        m1.train(); m2.train(); tot = 0.0
        for xi, xs, yb, lb in tl:
            xi, xs, yb, lb = xi.to(device), xs.to(device), yb.to(device), lb.to(device)
            xi = prep_img(xi, lb, None, aug, True)
            xs = drop_sensor(xs, p_drop)
            u1 = F.cross_entropy(m1(xi, xs), yb, reduction='none')
            u2 = F.cross_entropy(m2(xi, xs), yb, reduction='none')
            k = max(1, int(round((1 - rate) * len(yb))))
            pilih1 = torch.argsort(u1.detach())[:k]      # sampel terpilih oleh jaringan 1
            pilih2 = torch.argsort(u2.detach())[:k]      # sampel terpilih oleh jaringan 2
            w = wv[yb]
            loss1 = (u1 * w)[pilih2].mean()              # jaringan 1 belajar dari pilihan jaringan 2
            loss2 = (u2 * w)[pilih1].mean()              # jaringan 2 belajar dari pilihan jaringan 1
            o1.zero_grad(); loss1.backward(); o1.step(); s1.step()
            o2.zero_grad(); loss2.backward(); o2.step(); s2.step()
            tot += loss1.item() * len(yb)
        P = predict3(m1, vl, device)
        f1 = f1_score(y_eval[va_idx], P.argmax(1), average='macro')
        hist.append((tot / len(tr_idx), f1))
        print(f"ep {ep+1}: forget {rate:.3f} | loss {hist[-1][0]:.4f} | val macro-F1 (jaringan 1) {f1:.4f}")
    return hist
