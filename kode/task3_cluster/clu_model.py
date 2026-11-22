"""Model representasi tanpa label untuk Task 3: autoencoder gambar, SimCLR gambar,
autoencoder spektrum, dan autoencoder gabungan (fusi tingkat fitur dengan modality dropout)."""
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F


def atur_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ---------- gambar ----------
def _blok(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, stride=2, padding=1), nn.BatchNorm2d(o), nn.GELU(),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.GELU())


def _naik(i, o):
    return nn.Sequential(nn.ConvTranspose2d(i, o, 4, stride=2, padding=1), nn.BatchNorm2d(o), nn.GELU())


class EncGambar(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.f = nn.Sequential(_blok(1, 32), _blok(32, 64), _blok(64, 128), _blok(128, 256))  # 64 -> 4
        self.fc = nn.Linear(256 * 4 * 4, d)

    def forward(self, x):          # x: (B,1,64,64)
        return self.fc(self.f(x).flatten(1))


class DecGambar(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.fc = nn.Linear(d, 256 * 4 * 4)
        self.g = nn.Sequential(_naik(256, 128), _naik(128, 64), _naik(64, 32), _naik(32, 16),
                               nn.Conv2d(16, 1, 3, padding=1))

    def forward(self, z):
        return self.g(self.fc(z).view(-1, 256, 4, 4))


class AEGambar(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.enc, self.dec = EncGambar(d), DecGambar(d)

    def forward(self, x):
        return self.dec(self.enc(x))


class SimCLR(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.enc = EncGambar(d)
        self.proj = nn.Sequential(nn.GELU(), nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 64))

    def forward(self, x):
        return F.normalize(self.proj(self.enc(x)), dim=1)


def nt_xent(z1, z2, tau=0.2):
    n = z1.shape[0]
    z = torch.cat([z1, z2])
    sim = z @ z.t() / tau
    sim = sim.masked_fill(torch.eye(2 * n, dtype=torch.bool, device=z.device), -1e9)
    tgt = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, tgt)


def aug_foto(x, kuat=1.0):
    """Kecerlangan, kontras, dan derau (meniru perbedaan akuisisi). x: (B,1,H,W)."""
    B, d = x.shape[0], x.device
    c = 1 + (torch.rand(B, 1, 1, 1, device=d) - 0.5) * 0.4 * kuat
    b = (torch.rand(B, 1, 1, 1, device=d) - 0.5) * 0.4 * kuat
    return x * c + b + torch.randn_like(x) * 0.05 * kuat


def aug_geo(x, rot=15.0, skala=0.3, geser=0.2):
    """Rotasi kecil, skala, pergeseran, dan cermin horizontal acak."""
    B, d = x.shape[0], x.device
    ang = (torch.rand(B, device=d) - 0.5) * 2 * math.radians(rot)
    sc = 1 + (torch.rand(B, device=d) - 0.5) * skala
    tx = (torch.rand(B, device=d) - 0.5) * geser
    ty = (torch.rand(B, device=d) - 0.5) * geser
    flip = (torch.rand(B, device=d) < 0.5).float() * 2 - 1
    cs, sn = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cs * flip, -sn, tx], 1), torch.stack([sn * flip, cs, ty], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, padding_mode='reflection', align_corners=False)


# ---------- spektrum ----------
class EncSpektrum(nn.Module):
    def __init__(self, d=32, in_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, d))

    def forward(self, x):
        return self.net(x)


class DecSpektrum(nn.Module):
    def __init__(self, d=32, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 256), nn.GELU(), nn.Linear(256, out_dim))

    def forward(self, z):
        return self.net(z)


class AESpektrum(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.enc, self.dec = EncSpektrum(d), DecSpektrum(d)

    def forward(self, x):
        return self.dec(self.enc(x))


# ---------- gabungan ----------
class AEGabungan(nn.Module):
    """Dua encoder -> fusi tingkat fitur -> kode bersama z -> dua dekoder.
    Modality dropout mematikan salah satu modalitas secara acak saat pelatihan."""
    def __init__(self, d=64, h=128):
        super().__init__()
        self.eg, self.es = EncGambar(h), EncSpektrum(h)
        self.fuse = nn.Sequential(nn.GELU(), nn.Linear(2 * h, 128), nn.GELU(), nn.Linear(128, d))
        self.dg, self.ds = DecGambar(d), DecSpektrum(d)

    def encode(self, g, s, p_drop=0.0):
        hg, hs = self.eg(g), self.es(s)
        if self.training and p_drop > 0:
            r = torch.rand(hg.shape[0], 1, device=hg.device)
            hg = hg * (r >= p_drop / 2).float()                      # gambar dimatikan bila r < p/2
            hs = hs * ((r < p_drop / 2) | (r >= p_drop)).float()     # spektrum dimatikan bila p/2 <= r < p
        return self.fuse(torch.cat([hg, hs], 1))

    def decode(self, z):
        return self.dg(z), self.ds(z)


# ---------- pelatihan umum ----------
def latih(model, data, langkah, epoch, bs, lr, wd=1e-4, val_frac=0.1, seed=0, cetak=5, nama=''):
    """data: daftar tensor di GPU (dimensi pertama = sampel). langkah(model, *batch) -> loss."""
    atur_seed(seed)
    n = data[0].shape[0]
    dev = data[0].device
    perm = torch.randperm(n, device=dev)
    nv = int(n * val_frac)
    iv, it = perm[:nv], perm[nv:]
    spe = max(1, len(it) // bs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epoch * spe)
    riwayat = []
    for ep in range(1, epoch + 1):
        model.train()
        p = it[torch.randperm(len(it), device=dev)]
        tr = 0.0
        for i in range(spe):
            b = p[i * bs:(i + 1) * bs]
            loss = langkah(model, *[t[b] for t in data])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sch.step()
            tr += loss.item()
        model.eval()
        with torch.no_grad():
            va = float(np.mean([langkah(model, *[t[iv[j:j + 1024]] for t in data]).item()
                                for j in range(0, nv, 1024)]))
        riwayat.append((tr / spe, va))
        if ep == 1 or ep % cetak == 0 or ep == epoch:
            print(f"  {nama} ep {ep}/{epoch}: loss latih {tr / spe:.4f} | validasi {va:.4f}")
    return riwayat


@torch.no_grad()
def ekstrak(fn, data, bs=2048):
    """fn(*batch) -> embedding. Mengembalikan array numpy float32."""
    out = []
    for i in range(0, data[0].shape[0], bs):
        out.append(fn(*[t[i:i + bs] for t in data]).float().cpu())
    return torch.cat(out).numpy()


# ---------- fungsi langkah ----------
def langkah_ae_gambar(m, g):
    x = g.unsqueeze(1)
    return F.mse_loss(m(aug_foto(x, 0.5)), x)


def langkah_simclr(m, g):
    x = g.unsqueeze(1)
    v1, v2 = aug_foto(aug_geo(x), 1.0), aug_foto(aug_geo(x), 1.0)
    return nt_xent(m(v1), m(v2))


def langkah_ae_spektrum(m, s):
    return F.mse_loss(m(s + 0.1 * torch.randn_like(s)), s)


def langkah_gabungan(m, g, s):
    x = g.unsqueeze(1)
    z = m.encode(aug_foto(x, 0.5), s + 0.1 * torch.randn_like(s), p_drop=0.3)
    rg, rs = m.decode(z)
    return F.mse_loss(rg, x) + F.mse_loss(rs, s)
