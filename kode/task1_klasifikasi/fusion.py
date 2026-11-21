import numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score
from models import ImageCNN, SensorCNN
from train_utils import get_loader

def get_loader2(Xi, Xs, y, idx, bs=256, shuffle=False):
    """Loader dengan dua input: gambar, sensor, dan label."""
    ds = TensorDataset(torch.from_numpy(Xi[idx]), torch.from_numpy(Xs[idx]),
                       torch.from_numpy(y[idx]).long())
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)

class FusionNet(nn.Module):
    """Fusi tingkat fitur: vektor fitur gambar (128) + vektor fitur sensor (128) -> 256 -> kelas."""
    def __init__(self, n_class=7):
        super().__init__()
        self.img = ImageCNN(n_class)
        self.sen = SensorCNN(6, n_class)
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True),
                                  nn.Dropout(0.3), nn.Linear(128, n_class))
    def forward(self, xi, xs):
        return self.head(torch.cat([self.img.features(xi), self.sen.features(xs)], dim=1))

def drop_sensor(xs, p):
    """Modality dropout: dengan peluang p, seluruh sensor sebuah sampel dikosongkan
    (sinyal = 0 dan mask = 0), identik dengan representasi sensor yang tidak terekam."""
    if p <= 0:
        return xs
    keep = (torch.rand(xs.shape[0], 1, 1, device=xs.device) >= p).float()
    return xs * keep

@torch.no_grad()
def predict_fusion(model, loader, device, sensor_off=False, proba=False):
    model.eval(); out = []
    for xi, xs, _ in loader:
        xi, xs = xi.to(device), xs.to(device)
        if sensor_off:
            xs = torch.zeros_like(xs)
        logits = model(xi, xs)
        out.append(logits.softmax(1).cpu() if proba else logits.argmax(1).cpu())
    return torch.cat(out).numpy()

def fit_fusion(model, Xi, Xs, y, tr_idx, va_idx, epochs=15, lr=1e-3, bs=256,
               class_weight=None, p_drop=0.0, device='cuda', seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    model.to(device)
    w = None if class_weight is None else torch.tensor(class_weight, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(tr_idx) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    tl = get_loader2(Xi, Xs, y, tr_idx, bs, True)
    vl = get_loader2(Xi, Xs, y, va_idx, bs)
    hist = []
    for ep in range(epochs):
        model.train(); tot = 0.0
        for xi, xs, yb in tl:
            xi, xs, yb = xi.to(device), xs.to(device), yb.to(device)
            xs = drop_sensor(xs, p_drop)
            opt.zero_grad()
            loss = loss_fn(model(xi, xs), yb)
            loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(yb)
        f1 = f1_score(y[va_idx], predict_fusion(model, vl, device), average='macro')
        hist.append((tot / len(tr_idx), f1))
        print(f"ep {ep+1}: loss {hist[-1][0]:.4f} | val macro-F1 {f1:.4f}")
    return hist

@torch.no_grad()
def proba_single(model, X, y, idx, device, bs=512):
    """Probabilitas kelas dari model satu modalitas (untuk fusi keputusan)."""
    model.eval(); out = []
    for xb, _ in get_loader(X, y, idx, bs):
        out.append(model(xb.to(device)).softmax(1).cpu())
    return torch.cat(out).numpy()
