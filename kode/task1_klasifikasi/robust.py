import numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score
from fusion import drop_sensor

def lini_code(meta):
    return (meta['lini'].values == 'B').astype(np.int64)          # A=0, B=1

def lini_stats(img, lc, idx):
    # rerata dan simpangan baku piksel per lini, hanya dari data latih
    mu = np.zeros(2, np.float32); sd = np.ones(2, np.float32)
    for l in (0, 1):
        v = img[idx[lc[idx] == l]]
        mu[l], sd[l] = float(v.mean()), float(v.std())
    return mu, sd

def augment_img(xi, b_max=0.15, c_rng=(0.7, 1.3)):
    n = xi.shape[0]
    c = torch.empty(n, 1, 1, 1, device=xi.device).uniform_(*c_rng)
    b = torch.empty(n, 1, 1, 1, device=xi.device).uniform_(-b_max, b_max)
    m = xi.mean(dim=(1, 2, 3), keepdim=True)
    return ((xi - m) * c + m + b).clamp(0, 1)

def _norm_to(norm, device):
    if norm is None:
        return None
    return torch.tensor(norm[0], device=device), torch.tensor(norm[1], device=device)

def prep_img(xi, lb, nrm, aug, train):
    if aug and train:
        xi = augment_img(xi)
    if nrm is not None:
        mu, sd = nrm
        xi = (xi - mu[lb].view(-1, 1, 1, 1)) / sd[lb].view(-1, 1, 1, 1)
    return xi

def get_loader3(Xi, Xs, y, lc, idx, bs=256, shuffle=False):
    ds = TensorDataset(torch.from_numpy(Xi[idx]), torch.from_numpy(Xs[idx]),
                       torch.from_numpy(y[idx]).long(), torch.from_numpy(lc[idx]))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)

@torch.no_grad()
def predict3(model, loader, device, norm=None, sensor_off=False):
    model.eval(); nrm = _norm_to(norm, device); out = []
    for xi, xs, _, lb in loader:
        xi, xs, lb = xi.to(device), xs.to(device), lb.to(device)
        xi = prep_img(xi, lb, nrm, False, False)
        if sensor_off:
            xs = torch.zeros_like(xs)
        out.append(model(xi, xs).softmax(1).cpu())
    return torch.cat(out).numpy()

def fit3(model, Xi, Xs, y, lc, tr_idx, va_idx, epochs=15, lr=1e-3, bs=256, class_weight=None,
         p_drop=0.0, aug=False, norm=None, label_smoothing=0.0, device='cuda', seed=0, y_eval=None):
    torch.manual_seed(seed); np.random.seed(seed)
    y_eval = y if y_eval is None else y_eval
    model.to(device)
    w = None if class_weight is None else torch.tensor(class_weight, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=label_smoothing)
    nrm = _norm_to(norm, device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(tr_idx) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    tl = get_loader3(Xi, Xs, y, lc, tr_idx, bs, True)
    vl = get_loader3(Xi, Xs, y_eval, lc, va_idx)
    hist = []
    for ep in range(epochs):
        model.train(); tot = 0.0
        for xi, xs, yb, lb in tl:
            xi, xs, yb, lb = xi.to(device), xs.to(device), yb.to(device), lb.to(device)
            xi = prep_img(xi, lb, nrm, aug, True)
            xs = drop_sensor(xs, p_drop)
            opt.zero_grad()
            loss = loss_fn(model(xi, xs), yb)
            loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(yb)
        P = predict3(model, vl, device, norm=norm)
        f1 = f1_score(y_eval[va_idx], P.argmax(1), average='macro')
        hist.append((tot / len(tr_idx), f1))
        print(f"ep {ep+1}: loss {hist[-1][0]:.4f} | val macro-F1 {f1:.4f}")
    return hist
