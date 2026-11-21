"""Input fingerprints for numerical products. Presentation code never invalidates simulations."""
import ast
from functools import lru_cache
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()[:16]


# Each node lists upstream nodes, effective configuration fields and implementation inputs.
DEPENDENCIES = {
    "physics": ("", "physics prior_bounds", "physics.py simulator:stable_branch,interp_lambda sly4.dat"),
    "simulate": ("physics", "seed sim data", "data.py simulator.py"),
    "grid-build": ("physics", "grid.shape grid.m_lo grid.m_hi grid.dm", "grid.py"),
    "grid": ("grid-build simulate likelihood", "grid.n_validate", "stages:stage_grid"),
    "likelihood": ("grid-build", "sim", "likelihood.py simulator:chirp_mass,lambda_tilde"),
    "nested": ("likelihood", "dynesty", "reference.py metrics:kish_ess"),
    "train": ("simulate", "model train", "model.py train.py stages:Ctx.main_members"),
    "posterior": ("train", "sampling", "inference.py metrics.py stages:_summaries"),
    "evaluate": ("posterior", "evaluate derived speed ablation.n_eval run_derived_test", "stages:stage_evaluate,_speed,_derived_test,_by_n,_ridge_features,N_BINS"),
    "arch-model": ("train", "ablation", "stages:stage_ablate_arch"),
    "msg-model": ("train", "messenger", "stages:stage_ablate_messenger,MESSENGER_SETS,_nonempty"),
    "pristine-model": ("train", "zero_noise", "stages:pristine_catalog,stage_zero_noise"),
    "ablate-arch": ("arch-model posterior", "", ""),
    "ablate-messenger": ("msg-model posterior", "ablation.n_eval run_messenger_retrain", ""),
    "zero-noise": ("pristine-model posterior", "", "identifiability.py"),
    "identifiability": ("simulate", "fisher", "identifiability.py stages:stage_identifiability"),
    "convergence": ("simulate", "convergence", "stages:stage_convergence,_conv_job"),
    "catalog": ("", "real sim", "realdata.py simulator.py"),
    "realdata": ("posterior catalog", "derived", "stages:stage_realdata,_derive,_derived_table"),
    "reference-real": ("nested catalog", "", "stages:_ref_jobs_real"),
    "compare-real": ("realdata reference-real", "compare derived", "stages:stage_compare_real,_compare"),
    "reference-synth": ("nested posterior", "synth_ref compare", "stages:stage_reference_synth,_synth_catalogs,_compare"),
    "is-sampler": ("posterior likelihood", "importance", "importance.py reference.py"),
    "importance": ("is-sampler realdata reference-real reference-synth", "compare derived", "stages:stage_importance,_is,_derive,_derived_table,_compare"),
    "economics": ("compare-real importance simulate train grid-build", "economics run_subset_workflow", "stages:stage_economics"),
    "systematics": ("posterior", "", "stages:stage_systematics,_crust_job"),
    "summary": ("evaluate grid compare-real reference-synth importance economics convergence systematics identifiability zero-noise ablate-arch ablate-messenger", "", ""),
}

RESULT_STAGE = {
    "data_generation":"simulate", "norm_stats":"simulate", "grid_validation":"grid", "train_ensemble":"train",
    "eval_main":"evaluate", "ablation_arch":"ablate-arch", "ablation_messenger":"ablate-messenger",
    "compare_real":"compare-real", "reference_synth":"reference-synth", "paper_numbers":"summary", "timings":"summary",
}

FIGURE_STAGE = {
    "loss_main":"train", "calibration_main":"evaluate", "true_vs_pred_main":"evaluate", "mae_ratio_by_n":"evaluate",
    "sensitivity_scans":"identifiability", "real_corner_overlay":"compare-real", "fig1_mr_mlambda":"compare-real",
    "synthetic_reference_benchmark":"reference-synth", "is_efficiency_vs_N":"importance",
}


@lru_cache(maxsize=64)
def _file_digest(path, size, modified):
    h = hashlib.sha256()
    if path.suffix in {".py", ".dat"}:
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_digest(path):
    if not path.is_file():
        return None
    stat = path.stat()
    return _file_digest(path, stat.st_size, stat.st_mtime_ns)


def _sections(source):
    tree = ast.parse(source.read_text(encoding="utf-8"))
    sections = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            sections.update({f"{node.name}.{f.name}": ast.dump(f) for f in node.body if isinstance(f, ast.FunctionDef)})
        elif isinstance(node, ast.FunctionDef):
            sections[node.name] = ast.dump(node)
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            sections[node.targets[0].id] = ast.dump(node.value)
    return sections


class Fingerprints:
    def __init__(self, cfg, package=None):
        self.package = package or Path(__file__).parent
        self.cfg = cfg
        sections = {}
        self.static = {}
        for node, (_, settings, sources) in DEPENDENCIES.items():
            config = {}
            for key in settings.split():
                val = cfg
                for part in key.split("."):
                    val = val[part]
                config[key] = val
            code = {}
            for src in sources.split():
                if ":" in src:
                    module, names = src.split(":")
                    if module not in sections:
                        sections[module] = _sections(self.package / f"{module}.py")
                    for name in names.split(","):
                        code[f"{module}.{name}"] = sections[module][name]
                else:
                    code[src] = file_digest(self.package / src)
            self.static[node] = digest([node, config, code])

    def key(self, node):
        from .realdata import GW_SOURCES
        # A first run may download the observational files after Paths was constructed.
        def visit(name):
            upstream = DEPENDENCIES[name][0].split()
            inputs = [visit(k) for k in upstream]
            if name == "catalog":
                base = Path(self.cfg["root"]) / "data" / "gw_events"
                inputs.extend(file_digest(base / s["file"]) for s in GW_SOURCES.values())
            return digest([self.static[name], inputs])
        visit = lru_cache(None)(visit)
        return visit(node)
