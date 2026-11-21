"""Command-line entry point.

    python -m nseos.cli list
    python -m nseos.cli simulate train evaluate
    python -m nseos.cli all --root /content/drive/MyDrive/ns_eos_project
    python -m nseos.cli ablate-arch --set ablation.seeds=[0,1]
    python -m nseos.cli evaluate --force          # recompute this stage's cached outputs
"""
import argparse
import time

from .config import load_config


def _completion_key(ctx, stage):
    # Some reporting stages also consume optional files. A report run before
    # its upstream stage finishes must refresh when that input first appears.
    from pathlib import Path
    from .provenance import digest, file_digest
    from .stages import SUMMARY_INPUTS
    names = {"summary": SUMMARY_INPUTS, "economics": ("realdata", "compare_real", "importance"),
             "reference-real": ("realdata",), "compare-real": ("realdata",),
             "importance": ("realdata", "compare_real", "reference_synth")}.get(stage, ())
    items = [file_digest(Path(ctx.paths.result(name))) for name in names]
    if stage in ("summary", "economics"):
        items.append(ctx.timings.data)
    return digest([ctx.paths.tag_for(stage), items])


def main(argv=None):
    import sys
    for stream in (sys.stdout, sys.stderr):   # Windows consoles default to cp1252
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    from .stages import ORDER, STAGES, Ctx
    ap = argparse.ArgumentParser(prog="nseos")
    ap.add_argument("stages", nargs="+", help="stage names, 'all', or 'list'")
    ap.add_argument("--config", default=None, help="JSON file with config overrides")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. --set train.max_epochs=50")
    ap.add_argument("--root", default=None, help="artefact root directory (default $NSEOS_ROOT or ./ns_eos_project)")
    ap.add_argument("--force", action="store_true", help="recompute cached outputs of the requested stages")
    ap.add_argument("--log", default=None, help="also append all output to this file")
    a = ap.parse_args(argv)
    if a.log:
        class _Tee:
            def __init__(self, stream, fh):
                self.stream, self.fh = stream, fh

            def write(self, s):
                self.stream.write(s)
                self.fh.write(s)
                self.fh.flush()

            def flush(self):
                self.stream.flush()
                self.fh.flush()
        fh = open(a.log, "a", encoding="utf-8")
        sys.stdout, sys.stderr = _Tee(sys.stdout, fh), _Tee(sys.stderr, fh)
        print(f"\n##### run started {time.strftime('%Y-%m-%d %H:%M:%S')} : {' '.join(sys.argv)}")
    if a.stages == ["list"]:
        print("\n".join(ORDER))
        return
    overrides = list(a.set) + ([f"root={a.root}"] if a.root else [])
    cfg = load_config(a.config, overrides)
    names = ORDER if a.stages == ["all"] else a.stages
    unknown = [n for n in names if n not in STAGES]
    if unknown:
        ap.error(f"unknown stage(s) {unknown}; choose from {ORDER}")
    ctx = Ctx(cfg, force=a.force)
    # Completed stages leave a marker holding a fingerprint of the config. Re-running skips them
    # (instant resume after an interruption); a changed config or --force re-runs them, and the
    # stage's own caches still skip any finished heavy work.
    import os
    for n in names:
        marker = ctx.paths.result(f"stage_{n}", "done", stage=n)
        fp = _completion_key(ctx, n)
        if not a.force and os.path.exists(marker) and open(marker).read().strip() == fp:
            print(f"== stage {n}: already complete for this config (skipping; use --force to redo)", flush=True)
            continue
        t0 = time.time()
        print(f"\n{'=' * 70}\n== stage: {n}\n{'=' * 70}", flush=True)
        STAGES[n](ctx)
        # Downloads may have materialized catalog inputs during this stage.
        with open(ctx.paths.result(f"stage_{n}", "done", stage=n), "w") as fh:
            fh.write(_completion_key(ctx, n))
        print(f"== {n} done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
