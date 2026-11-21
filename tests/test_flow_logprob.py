"""The flow's log-density (needed for importance sampling) must be correct:
  1. forward sampling with log q and backward log_prob agree (round trip);
  2. q integrates to 1 (importance-sampling estimate with a broad Gaussian proposal);
  3. the ensemble mixture density is the average of member densities.
Requires torch (CPU is fine)."""
import math
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nseos.model import (EnsembleModel, MultiMessengerEOSModel, collate_fn, EOSCatalogDataset,  # noqa: E402
                         flow_log_prob, flow_sample_with_log_prob, model_log_prob)

STATS = {**{f"{t}_{k}": np.zeros(d) if "mu" in k else np.ones(d)
            for t, d in (("gw", 2), ("nicer", 2), ("pulsar", 1)) for k in ("x_mu", "x_sd", "s_mu", "s_sd")},
         "theta_mu": np.array([1.5, -0.2, 0.0, 0.0]), "theta_sd": np.array([0.3, 0.7, 0.35, 0.012])}


def _model(seed):
    torch.manual_seed(seed)
    m = MultiMessengerEOSModel(d_embed=16, d_hidden_flow=32, time_embed_dim=8, n_heads=2, n_attn_layers=1,
                               n_res_blocks=1).set_stats(STATS).eval()
    with torch.no_grad():   # make the velocity field non-trivial
        for p in m.vel_proj_out.parameters():
            p.mul_(3.0)
    return m


def _batch():
    cat = [{"type": "nicer", "x": [1.4, 12.0], "sigma": [0.1, 0.5]}, {"type": "pulsar", "x": [2.0], "sigma": [0.05]}]
    return collate_fn([EOSCatalogDataset([{"theta_eos": [0] * 4, "catalog": cat}])[0]])


def test_round_trip():
    m, b = _model(0), _batch()
    x, lq_fwd = flow_sample_with_log_prob(m, b, 500, n_steps=100, generator=torch.Generator().manual_seed(0))
    lq_bwd = flow_log_prob(m, b, x, n_steps=100)
    assert torch.max(torch.abs(lq_fwd - lq_bwd)).item() < 2e-3


def test_normalisation():
    m, b = _model(1), _batch()
    # Gaussian proposal fitted to flow samples and widened 1.5x -> low-variance IS estimate of ∫q
    ref = m.sample(b, 5000, 50, generator=torch.Generator().manual_seed(5))[0].double()
    prop = torch.distributions.MultivariateNormal(ref.mean(0), covariance_matrix=2.25 * torch.cov(ref.T))
    theta = prop.sample((100000,)).float().unsqueeze(0)
    lq = flow_log_prob(m, b, theta, n_steps=50).double()
    w = torch.exp(lq - prop.log_prob(theta[0].double()))
    est, se = w.mean().item(), (w.std() / math.sqrt(w.numel())).item()
    assert abs(est - 1) < max(0.03, 4 * se), (est, se)


def test_ensemble_mixture():
    ms, b = [_model(2), _model(3)], _batch()
    ens = EnsembleModel(ms).eval()
    th = ens.sample(b, 64, 20, generator=torch.Generator().manual_seed(2))
    lp = model_log_prob(ens, b, th, 20)
    ref = torch.log(0.5 * (torch.exp(flow_log_prob(ms[0], b, th, 20)) + torch.exp(flow_log_prob(ms[1], b, th, 20))))
    assert torch.allclose(lp, ref, atol=1e-4)
