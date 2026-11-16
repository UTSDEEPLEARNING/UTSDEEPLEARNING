import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import TensorDataset, DataLoader
from data import KELAS

def get_loader(X, y, idx, bs=256, shuffle=False):
    ds = TensorDataset(torch.from_numpy(X[idx]), torch.from_numpy(y[idx]).long())
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)

@torch.no_grad()
def predict(model, loader, device):
    model.eval(); out = []
    for xb, _ in loader:
        out.append(model(xb.to(device)).argmax(1).cpu())
    return torch.cat(out).numpy()

def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

def fit(model, X, y, tr_idx, va_idx, epochs=10, lr=1e-3, bs=256,
        class_weight=None, device='cuda', seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    model.to(device)
    w = None if class_weight is None else torch.tensor(class_weight, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(tr_idx) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    tl, vl = get_loader(X, y, tr_idx, bs, True), get_loader(X, y, va_idx, bs)
    hist = []
    for ep in range(epochs):
        model.train(); tot = 0.0
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(xb)
        f1 = macro_f1(y[va_idx], predict(model, vl, device))
        hist.append((tot / len(tr_idx), f1))
        print(f"ep {ep+1}: loss {hist[-1][0]:.4f} | val macro-F1 {f1:.4f}")
    return hist

def overfit_one_batch(model, X, y, idx, n=64, steps=200, lr=1e-3, device='cuda'):
    model.to(device).train()
    xb = torch.from_numpy(X[idx[:n]]).to(device)
    yb = torch.from_numpy(y[idx[:n]]).long().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for s in range(steps):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(xb), yb)
        loss.backward(); opt.step()
        if s % 50 == 0 or s == steps - 1:
            print(s, round(loss.item(), 4))

def report(y_true, y_pred, lini):
    print("macro-F1 total:", round(macro_f1(y_true, y_pred), 4))
    for l in ['A', 'B']:
        m = (lini == l)
        print(f"lini {l}: macro-F1 {macro_f1(y_true[m], y_pred[m]):.4f}")
    print(classification_report(y_true, y_pred, target_names=KELAS, digits=3))
