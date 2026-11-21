"""Torch dataset, collate and the amortised posterior model.

Fixes / additions relative to v1:
  * sigma enters the encoders as standardised log(sigma); padded slots are zeroed
    before the encoders, so masked garbage can never produce NaNs (B1).
  * ``aggregator`` in {set_transformer, deepsets, padded_mlp} with an *identical*
    flow-matching decoder -> controlled architecture ablation (reviewers DVmA/VWKG/HFbp).
  * theta normalisation lives in model buffers, so a checkpoint is self-contained.
  * ``augment=True`` re-simulates the whole catalog from the stored EOS curve every epoch.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .simulator import OBS_DIMS, OBS_TYPES, SIM_CFG, simulate_catalog


class EOSCatalogDataset(Dataset):
    def __init__(self, samples, augment=False, allowed_types=None, sim_cfg=SIM_CFG, base_seed=0,
                 catalog_fn=None):
        self.samples, self.augment, self.allowed_types = samples, augment, allowed_types
        self.sim_cfg, self.base_seed, self.catalog_fn = sim_cfg, base_seed, catalog_fn
        self._epoch = 0
        self.shared_epoch = None

    @property
    def epoch(self):
        return self._epoch if self.shared_epoch is None else int(self.shared_epoch.item())

    @epoch.setter
    def epoch(self, value):
        self._epoch = int(value)
        if self.shared_epoch is not None:
            self.shared_epoch.fill_(value)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        if self.augment and "curve" in s:
            rng = np.random.default_rng([self.base_seed, self.epoch, s.get("idx", idx)])
            branch = tuple(s["curve"])
            if self.catalog_fn is not None:
                cat = self.catalog_fn(branch, rng)
            else:
                cat = simulate_catalog(branch, rng, self.sim_cfg, allowed_types=self.allowed_types)
        else:
            cat = s["catalog"]
            if self.allowed_types is not None:
                cat = [o for o in cat if o["type"] in self.allowed_types]
        return {"theta": torch.tensor(s["theta_eos"], dtype=torch.float32),
                "catalog": [{"type": o["type"], "x": torch.tensor(o["x"], dtype=torch.float32),
                             "sigma": torch.tensor(o["sigma"], dtype=torch.float32)} for o in cat]}


def collate_fn(batch):
    B = len(batch)
    out = {"theta": torch.stack([b["theta"] for b in batch]), "num_catalogs": B}
    for t, d in OBS_DIMS.items():
        n_max = max((sum(1 for o in b["catalog"] if o["type"] == t) for b in batch), default=0)
        x, s = torch.zeros(B, n_max, d), torch.zeros(B, n_max, d)
        mask = torch.ones(B, n_max, dtype=torch.bool)  # True = padding
        for i, b in enumerate(batch):
            j = 0
            for o in b["catalog"]:
                if o["type"] == t:
                    x[i, j], s[i, j], mask[i, j] = o["x"], o["sigma"], False
                    j += 1
        out[f"{t}_x"], out[f"{t}_sigma"], out[f"{t}_mask"] = x, s, mask
    return out


def move_to_device(batch, dev):
    return {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000) / (half - 1)))
        emb = t[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, x):
        return x + self.net(x)


AGGREGATORS = ("set_transformer", "deepsets", "padded_mlp")


class MultiMessengerEOSModel(nn.Module):
    def __init__(self, d_embed=256, d_hidden_flow=512, time_embed_dim=64, theta_dim=4,
                 aggregator="set_transformer", n_heads=8, n_attn_layers=6, n_res_blocks=4,
                 max_slots=SIM_CFG["N_MAX"]):
        super().__init__()
        assert aggregator in AGGREGATORS, aggregator
        self.aggregator, self.d_embed, self.theta_dim, self.max_slots = aggregator, d_embed, theta_dim, max_slots
        for t, d in OBS_DIMS.items():
            self.register_buffer(f"{t}_x_mu", torch.zeros(d)); self.register_buffer(f"{t}_x_sd", torch.ones(d))
            self.register_buffer(f"{t}_s_mu", torch.zeros(d)); self.register_buffer(f"{t}_s_sd", torch.ones(d))
        self.register_buffer("theta_mu", torch.zeros(theta_dim))
        self.register_buffer("theta_sd", torch.ones(theta_dim))

        def enc(in_dim):
            return nn.Sequential(nn.Linear(in_dim, d_embed), nn.GELU(), nn.LayerNorm(d_embed),
                                 nn.Linear(d_embed, d_embed), nn.GELU(), nn.LayerNorm(d_embed),
                                 nn.Linear(d_embed, d_embed))
        if aggregator in ("set_transformer", "deepsets"):
            self.encoders = nn.ModuleDict({t: enc(2 * d) for t, d in OBS_DIMS.items()})
            self.type_emb = nn.ParameterDict({t: nn.Parameter(torch.randn(1, d_embed) * 0.02) for t in OBS_TYPES})
        if aggregator == "set_transformer":
            layer = nn.TransformerEncoderLayer(d_model=d_embed, nhead=n_heads, dim_feedforward=4 * d_embed,
                                               batch_first=True, dropout=0.1)
            self.set_tf = nn.TransformerEncoder(layer, num_layers=n_attn_layers, enable_nested_tensor=False)
            self.pool_q = nn.Parameter(torch.randn(1, 1, d_embed) * 0.02)
            self.pool_attn = nn.MultiheadAttention(d_embed, n_heads, batch_first=True)
            self.pool_norm = nn.LayerNorm(d_embed)
        if aggregator == "padded_mlp":
            in_dim = sum(max_slots * (2 * d + 1) for d in OBS_DIMS.values())
            self.padded_mlp = nn.Sequential(nn.Linear(in_dim, 1024), nn.GELU(), nn.LayerNorm(1024),
                                            nn.Linear(1024, 512), nn.GELU(), nn.LayerNorm(512),
                                            nn.Linear(512, d_embed))
        self.time_emb = SinusoidalTimeEmbedding(time_embed_dim)
        self.vel_proj_in = nn.Linear(theta_dim + time_embed_dim + d_embed + 1, d_hidden_flow)
        self.vel_res_blocks = nn.Sequential(*[ResidualBlock(d_hidden_flow) for _ in range(n_res_blocks)])
        self.vel_proj_out = nn.Sequential(nn.LayerNorm(d_hidden_flow), nn.Linear(d_hidden_flow, theta_dim))
        self.last_nfe = 0

    # ── normalisation ──
    def set_stats(self, stats):
        with torch.no_grad():
            for t in OBS_TYPES:
                for a, b in (("x_mu", "x_mu"), ("x_sd", "x_sd"), ("s_mu", "s_mu"), ("s_sd", "s_sd")):
                    getattr(self, f"{t}_{a}").copy_(torch.as_tensor(np.asarray(stats[f"{t}_{b}"]), dtype=torch.float32))
            self.theta_mu.copy_(torch.as_tensor(np.asarray(stats["theta_mu"]), dtype=torch.float32))
            self.theta_sd.copy_(torch.as_tensor(np.asarray(stats["theta_sd"]), dtype=torch.float32))
        return self

    def normalize_theta(self, theta):
        return (theta - self.theta_mu) / self.theta_sd

    def denormalize_theta(self, z):
        return z * self.theta_sd + self.theta_mu

    def _features(self, batch, t, dev):
        x, sig, mask = batch[f"{t}_x"].to(dev), batch[f"{t}_sigma"].to(dev), batch[f"{t}_mask"].to(dev)
        x = (x - getattr(self, f"{t}_x_mu")) / getattr(self, f"{t}_x_sd")
        ls = (torch.log(sig.clamp_min(1e-12)) - getattr(self, f"{t}_s_mu")) / getattr(self, f"{t}_s_sd")
        pad = mask.unsqueeze(-1)
        x = torch.where(pad, torch.zeros_like(x), x)
        ls = torch.where(pad, torch.zeros_like(ls), ls)
        return x, ls, mask

    # ── aggregation ──
    def _count(self, batch, dev):
        return sum((~batch[f"{t}_mask"].to(dev)).float().sum(1, keepdim=True) for t in OBS_TYPES)

    def _tokens(self, batch, dev):
        embs, masks = [], []
        for t in OBS_TYPES:
            if batch[f"{t}_x"].shape[1] == 0:
                continue
            x, ls, mask = self._features(batch, t, dev)
            embs.append(self.encoders[t](torch.cat([x, ls], -1)).float() + self.type_emb[t])
            masks.append(mask)
        return embs, masks

    def encode_and_aggregate(self, batch):
        dev = self.theta_mu.device
        B = batch["num_catalogs"]
        cnt = self._count(batch, dev)
        if self.aggregator == "padded_mlp":
            feats = []
            for t in OBS_TYPES:
                x, ls, mask = self._features(batch, t, dev)
                f = torch.cat([x, ls, (~mask).float().unsqueeze(-1)], -1)
                if f.shape[1] > self.max_slots:
                    raise ValueError(f"padded_mlp supports at most {self.max_slots} {t} observations")
                feats.append(F.pad(f, (0, 0, 0, self.max_slots - f.shape[1])).reshape(B, -1))
            pooled = self.padded_mlp(torch.cat(feats, -1)).float()
        else:
            embs, masks = self._tokens(batch, dev)
            if not embs:
                pooled = torch.zeros(B, self.d_embed, device=dev)
            elif self.aggregator == "deepsets":
                acc = sum((e * (~m).float().unsqueeze(-1)).sum(1) for e, m in zip(embs, masks))
                pooled = acc / cnt.clamp(min=1)
            else:
                tok, full_mask = torch.cat(embs, 1), torch.cat(masks, 1)
                h = self.set_tf(tok, src_key_padding_mask=full_mask)
                q = self.pool_q.expand(B, -1, -1)
                p, _ = self.pool_attn(q, h, h, key_padding_mask=full_mask)
                pooled = self.pool_norm(p + q).squeeze(1)
        return torch.cat([pooled, torch.log1p(cnt)], -1)

    # ── flow matching ──
    def _velocity(self, x_t, t_emb, summary):
        return self.vel_proj_out(self.vel_res_blocks(self.vel_proj_in(torch.cat([x_t, t_emb, summary], -1))))

    def compute_loss(self, batch, theta, generator=None):
        """theta in physical units."""
        theta_n = self.normalize_theta(theta.to(self.theta_mu.device))
        summary = self.encode_and_aggregate(batch)
        B, dev = summary.shape[0], summary.device
        t = torch.rand(B, 1, device=dev, generator=generator)
        z = torch.randn(B, self.theta_dim, device=dev, generator=generator)
        v = self._velocity((1 - t) * z + t * theta_n, self.time_emb(t.squeeze(-1)), summary)
        return F.mse_loss(v, theta_n - z)

    @torch.no_grad()
    def sample(self, batch, n_samples=1000, n_steps=50, generator=None):
        """Physical-unit samples (B, n, 4): one aggregator pass + RK4 with 4*n_steps
        velocity-network evaluations (self.last_nfe)."""
        summary = self.encode_and_aggregate(batch)
        B, dev = summary.shape[0], summary.device
        s_rep = summary.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)
        z = torch.randn(B * n_samples, self.theta_dim, device=dev, generator=generator)
        dt = 1.0 / n_steps

        def vel(zv, tv):
            return self._velocity(zv, self.time_emb(torch.full((zv.shape[0],), tv, device=dev)), s_rep)
        for k in range(n_steps):
            t = k * dt
            k1 = vel(z, t); k2 = vel(z + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = vel(z + 0.5 * dt * k2, t + 0.5 * dt); k4 = vel(z + dt * k3, t + dt)
            z = z + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.last_nfe = 4 * n_steps
        return self.denormalize_theta(z).reshape(B, n_samples, self.theta_dim)


LOG_2PI = math.log(2 * math.pi)


def _vel_div(model, z, t, s_rep):
    """Velocity and its exact divergence tr(dv/dz) (theta_dim backward passes; cheap for 4-D)."""
    with torch.enable_grad():
        z = z.detach().requires_grad_(True)
        v = model._velocity(z, model.time_emb(torch.full((z.shape[0],), float(t), device=z.device)), s_rep)
        div = torch.zeros(z.shape[0], device=z.device)
        for i in range(model.theta_dim):
            g = torch.autograd.grad(v[:, i].sum(), z, retain_graph=i < model.theta_dim - 1)[0]
            div = div + g[:, i]
    return v.detach(), div.detach()


def _integrate_with_div(model, z, s_rep, t0, t1, n_steps):
    """RK4 on the augmented state (z, ∫ tr(dv/dz) dt) from t0 to t1."""
    dt = (t1 - t0) / n_steps
    acc = torch.zeros(z.shape[0], device=z.device)
    for k in range(n_steps):
        t = t0 + k * dt
        v1, d1 = _vel_div(model, z, t, s_rep)
        v2, d2 = _vel_div(model, z + 0.5 * dt * v1, t + 0.5 * dt, s_rep)
        v3, d3 = _vel_div(model, z + 0.5 * dt * v2, t + 0.5 * dt, s_rep)
        v4, d4 = _vel_div(model, z + dt * v3, t + dt, s_rep)
        z = z + dt / 6.0 * (v1 + 2 * v2 + 2 * v3 + v4)
        acc = acc + dt / 6.0 * (d1 + 2 * d2 + 2 * d3 + d4)
    return z, acc


def _std_normal_logpdf(z):
    return -0.5 * (z ** 2).sum(-1) - 0.5 * z.shape[-1] * LOG_2PI


@torch.no_grad()
def flow_log_prob(model, batch, theta, n_steps=50, chunk=20000):
    """log q(theta | catalog) in *physical* units for theta (B, n, 4), by integrating the
    probability-flow ODE backwards (continuous change of variables, exact trace)."""
    summary = model.encode_and_aggregate(batch)
    B, n, d = theta.shape
    x = model.normalize_theta(theta.to(summary.device)).reshape(B * n, d)
    s_rep = summary.unsqueeze(1).expand(B, n, -1).reshape(B * n, -1)
    out = []
    for i in range(0, B * n, chunk):
        z0, acc = _integrate_with_div(model, x[i:i + chunk], s_rep[i:i + chunk], 1.0, 0.0, n_steps)
        out.append(_std_normal_logpdf(z0) + acc)          # acc = ∫_1^0 tr dt = -∫_0^1 tr dt
    return (torch.cat(out) - torch.log(model.theta_sd).sum()).reshape(B, n)


@torch.no_grad()
def flow_sample_with_log_prob(model, batch, n_samples, n_steps=50, generator=None):
    """Forward sampling that also returns log q (used to test flow_log_prob by round trip)."""
    summary = model.encode_and_aggregate(batch)
    B = summary.shape[0]
    s_rep = summary.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)
    z0 = torch.randn(B * n_samples, model.theta_dim, device=summary.device, generator=generator)
    x, acc = _integrate_with_div(model, z0, s_rep, 0.0, 1.0, n_steps)
    logq = _std_normal_logpdf(z0) - acc - torch.log(model.theta_sd).sum()
    return model.denormalize_theta(x).reshape(B, n_samples, -1), logq.reshape(B, n_samples)


class EnsembleModel(nn.Module):
    """Equal-weight mixture of independently seeded flows (fix: seed variance / single-model
    headline numbers). Exposes the same interface as MultiMessengerEOSModel."""

    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.theta_dim = members[0].theta_dim
        self.last_nfe = 0

    @property
    def theta_mu(self):
        return self.members[0].theta_mu

    def encode_and_aggregate(self, batch):
        return self.members[0].encode_and_aggregate(batch)

    @torch.no_grad()
    def sample(self, batch, n_samples=1000, n_steps=50, generator=None):
        K = len(self.members)
        counts = [n_samples // K + (1 if k < n_samples % K else 0) for k in range(K)]
        outs = [m.sample(batch, c, n_steps, generator) for m, c in zip(self.members, counts) if c > 0]
        x = torch.cat(outs, 1)
        perm = torch.randperm(n_samples, device=x.device, generator=generator)
        self.last_nfe = 4 * n_steps
        return x[:, perm]

    @torch.no_grad()
    def log_prob(self, batch, theta, n_steps=50):
        lp = torch.stack([flow_log_prob(m, batch, theta, n_steps) for m in self.members])
        return torch.logsumexp(lp, 0) - math.log(len(self.members))


def model_log_prob(model, batch, theta, n_steps=50):
    if isinstance(model, EnsembleModel):
        return model.log_prob(batch, theta, n_steps)
    return flow_log_prob(model, batch, theta, n_steps)


def build_model(cfg, stats, aggregator="set_transformer", seed=None, device="cpu"):
    if seed is not None:
        torch.manual_seed(seed)
    m = MultiMessengerEOSModel(aggregator=aggregator, max_slots=cfg["sim"]["N_MAX"], **cfg["model"])
    return m.set_stats(stats).to(device)


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
