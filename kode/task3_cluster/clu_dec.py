"""Deep Embedded Clustering (DEC) di atas embedding tetap dari tahap 1.
Alur: autoencoder MLP pada embedding -> K-Means pada kode laten -> penyulingan dengan KL(P||Q)."""
import copy, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.cluster import KMeans


def atur_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class AEEmb(nn.Module):
    def __init__(self, d_in, d_z=16, h=256):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, 128), nn.ReLU(), nn.Linear(128, d_z))
        self.dec = nn.Sequential(nn.Linear(d_z, 128), nn.ReLU(), nn.Linear(128, h), nn.ReLU(), nn.Linear(h, d_in))

    def forward(self, x):
        z = self.enc(x)
        return z, self.dec(z)


def latih_ae(x, d_z=16, epoch=60, bs=256, lr=1e-3, noise=0.1, seed=0):
    """x: tensor (N,D) di GPU, sudah distandarkan per dimensi."""
    atur_seed(seed)
    ae = AEEmb(x.shape[1], d_z).to(x.device)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    n = x.shape[0]
    riwayat = []
    for ep in range(epoch):
        ae.train()
        perm = torch.randperm(n, device=x.device)
        tot, m = 0.0, 0
        for i in range(0, n - bs + 1, bs):
            xb = x[perm[i:i + bs]]
            _, rec = ae(xb + noise * torch.randn_like(xb))
            loss = F.mse_loss(rec, xb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item(); m += 1
        riwayat.append(tot / max(m, 1))
    return ae, riwayat


@torch.no_grad()
def kode(ae, x, bs=4096):
    ae.eval()
    return torch.cat([ae.enc(x[i:i + bs]) for i in range(0, x.shape[0], bs)])


def soft_assign(z, mu, alpha=1.0):
    d2 = ((z.unsqueeze(1) - mu.unsqueeze(0)) ** 2).sum(-1)
    q = (1.0 + d2 / alpha) ** (-(alpha + 1.0) / 2.0)
    return q / q.sum(1, keepdim=True)


def target(q):
    w = q ** 2 / q.sum(0, keepdim=True)
    return w / w.sum(1, keepdim=True)


def init_kmeans(z, k, seed=0, n_init=20):
    km = KMeans(k, n_init=n_init, random_state=seed).fit(z.cpu().numpy())
    return km.cluster_centers_.astype(np.float32), km.labels_


def dec(x, ae0, k, gamma=0.0, rounds=40, lr=1e-3, bs=256, tol=0.001, seed=0):
    """Menjalankan DEC dari autoencoder terlatih ae0 (tidak diubah; disalin).
    gamma > 0 menambahkan regulasi rekonstruksi (gaya IDEC). Mengembalikan (label, kode, riwayat_perubahan, label_init)."""
    atur_seed(seed)
    ae = copy.deepcopy(ae0)
    mu0, lab_init = init_kmeans(kode(ae, x), k, seed)
    mu = nn.Parameter(torch.tensor(mu0, device=x.device))
    params = list(ae.enc.parameters()) + [mu] + (list(ae.dec.parameters()) if gamma > 0 else [])
    opt = torch.optim.Adam(params, lr=lr)
    n = x.shape[0]
    prev, riwayat = None, []
    for r in range(rounds):
        with torch.no_grad():
            z = kode(ae, x)
            q = soft_assign(z, mu)
            p = target(q)
            lab = q.argmax(1)
        if prev is not None:
            delta = (lab != prev).float().mean().item()
            riwayat.append(delta)
            if delta < tol:
                break
        prev = lab
        ae.train()
        perm = torch.randperm(n, device=x.device)
        for i in range(0, n - bs + 1, bs):
            b = perm[i:i + bs]
            zb = ae.enc(x[b])
            qb = soft_assign(zb, mu).clamp_min(1e-8)
            loss = F.kl_div(qb.log(), p[b], reduction='batchmean')
            if gamma > 0:
                loss = loss + gamma * F.mse_loss(ae.dec(zb), x[b])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad():
        z = kode(ae, x)
        lab = soft_assign(z, mu).argmax(1)
    return lab.cpu().numpy(), z.cpu().numpy(), riwayat, lab_init
