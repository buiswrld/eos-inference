"""Central configuration. Every stage reads the same config, so the simulator,
the likelihood and the network always see the same forward model.

Override with a JSON file (``--config my.json``) or ``--set key.sub=value``.
"""
import copy
from .provenance import Fingerprints, RESULT_STAGE, FIGURE_STAGE, digest
import json
import os

DEFAULT_CONFIG = {
    "run_tag": "sly4",
    "seed": 42,
    # optional heavy sub-steps
    "run_derived_test": True,       # evaluate: TOV on posterior draws of 100 test catalogs
    "run_messenger_retrain": True,  # ablate-messenger: retrain one model per subset
    "run_subset_workflow": True,    # economics: all 2^k-1 subsets of the real catalog
    # Root for all artefacts (data, runs, results, figures). On Colab point this at Drive.
    "root": os.environ.get("NSEOS_ROOT", os.path.abspath("ns_eos_project")),
    "prior_bounds": {
        "gamma0": [0.2, 2.0],
        "gamma1": [-1.6, 1.7],
        "gamma2": [-0.6, 0.6],
        "gamma3": [-0.02, 0.02],
    },
    "physics": {"n_pc": 30},
    "sim": {
        "N_MIN": 1, "N_MAX": 50, "DIRICHLET_ALPHA": [1.0, 1.0, 1.0],
        "GW_M_LO": 1.1, "GW_M_HI": 2.0, "GW_SIG_LT": [40.0, 800.0], "GW_SIG_MC": [5e-4, 3e-2],
        "NICER_M_LO": 1.2, "NICER_M_HI": 2.2, "NICER_SIG_M": [0.02, 0.20], "NICER_SIG_R": [0.3, 1.6],
        "PSR_M_LO": 1.4, "PSR_M_HI": 2.5, "PSR_SIG_M": [0.01, 0.25],
        "MMAX_MIN": 1.5, "BRANCH_MMIN_MAX": 1.0,   # v1 cut at 2.0 double-counted the pulsars
    },
    "data": {"n_samples": 100000, "attempt_factor": 4, "split": [0.70, 0.15, 0.15], "n_probe": 1000},
    "model": {"d_embed": 256, "d_hidden_flow": 512, "time_embed_dim": 64, "n_heads": 8,
              "n_attn_layers": 6, "n_res_blocks": 4},
    "train": {"batch_size": 256, "num_workers": 8, "lr": 3e-4, "weight_decay": 1e-4,
              "max_epochs": 400, "patience": 40, "compile": False,
              "ensemble_seeds": [0, 1, 2]},
    "sampling": {"n_steps": 50, "n_eval_samples": 1000, "n_real_samples": 10000},
    "ablation": {"archs": ["set_transformer", "deepsets", "padded_mlp"], "seeds": [0, 1, 2],
                 "max_epochs": 150, "patience": 20, "n_eval": 4000},
    "messenger": {"max_epochs": 150, "patience": 20},
    "zero_noise": {"n_train": 20000, "n_test": 2000, "max_epochs": 300, "patience": 30,
                   "n_nicer": 16, "n_gw": 8, "sig_lt": 1.0, "sig_mc": 1e-4, "sig_m": 1e-3, "sig_r": 0.01},
    "fisher": {"n_theta": 150, "n_pc": 60, "step_frac": 0.1,
               "masses": [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
               "scenarios": {"current": [1.0, 0.5, 0.10], "next_gen": [0.3, 0.15, 0.03],
                             "near_zero": [0.03, 0.015, 0.003]}},
    "grid": {"shape": [20, 20, 10, 5], "m_lo": 1.0, "m_hi": 2.5, "dm": 0.05, "n_validate": 1000, "block": 1000},
    "dynesty": {"nlive": 500, "dlogz": 0.1, "sample": "rwalk", "bound": "multi", "maxcall": 5000000,
                "checkpoint_every": 120},
    "real": {"policy": "dedup", "download_gw": True},
    "synth_ref": {"n_per_bin": 8, "bins": [[1, 4], [5, 10], [11, 20]], "n_floor": 6, "n_sbi_samples": 10000},
    "compare": {"n_subsample_repeats": 200, "kde_grid": 400, "n_band_curves": 300},
    "evaluate": {"nested_n_theta": 300, "nested_N": [1, 3, 5, 10, 20, 30, 50], "tarp_n": 3000},
    "speed": {"n_list": [1000, 10000], "steps": [10, 25, 50], "throughput_catalogs": 256,
              "conv_catalogs": 128, "conv_ref_steps": 100},
    "economics": {"subset_n_samples": 2000, "loo_n_samples": 300},
    "derived": {"n_test_catalogs": 100, "n_samples_per_catalog": 100, "n_real": 2000},
    "importance": {"n_real": 20000, "n_synth": 20000, "n_test_catalogs": 100, "n_test": 5000, "defensive": 0.1},
    "convergence": {"n_theta": 100, "n_pc_hi": 120, "rtol_hi": 1e-8},
}


def _set_path(cfg, dotted, value):
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    try:
        value = json.loads(value)
    except (TypeError, ValueError):
        pass
    d[keys[-1]] = value


def _merge(base, upd):
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path=None, overrides=()):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        with open(path) as fh:
            _merge(cfg, json.load(fh))
    for ov in overrides or ():
        k, _, v = ov.partition("=")
        _set_path(cfg, k.strip(), v.strip())
    return cfg


class Paths:
    """All artefact locations derived from cfg['root'] and cfg['run_tag']."""

    def __init__(self, cfg):
        self.root = cfg["root"]
        self.fingerprints = Fingerprints(cfg)
        self.prefix = cfg["run_tag"]
        for d in ("data", "runs", "results", "figures", "reference", "grid"):
            os.makedirs(os.path.join(self.root, d), exist_ok=True)

    def tag_for(self, stage, inputs=None):
        key = self.fingerprints.key(stage)
        return f"{self.prefix}_{key if inputs is None else digest([key, inputs])}"

    @property
    def tag(self):
        return self.tag_for("summary")

    def data(self, name, stage="simulate", inputs=None):
        return os.path.join(self.root, "data", f"{name}_{self.tag_for(stage, inputs)}")

    def run(self, run_name):
        stage = "train"
        if run_name.startswith("arch_"):
            stage = "arch-model"
        elif run_name.startswith("msg_"):
            stage = "msg-model"
        elif run_name == "zero_noise":
            stage = "pristine-model"
        p = os.path.join(self.root, "runs", f"{run_name}_{self.tag_for(stage)}")
        os.makedirs(p, exist_ok=True)
        return p

    def result(self, name, ext="json", stage=None):
        stage = stage or RESULT_STAGE.get(name, name.replace("_", "-"))
        return os.path.join(self.root, "results", f"{name}_{self.tag_for(stage)}.{ext}")

    def figure(self, name):
        return os.path.join(self.root, "figures", f"{name}_{self.tag_for(FIGURE_STAGE[name])}")

    def reference(self, name, catalog):
        obs = [(o["type"], o["x"], o["sigma"]) for o in catalog]
        return os.path.join(self.root, "reference", f"{name}_{self.tag_for('nested', obs)}.pkl")

    def grid(self):
        return os.path.join(self.root, "grid", f"tov_grid_{self.tag_for('grid-build')}.pkl")
