"""Regenerate notebooks/colab_driver.ipynb (run from the repo root)."""
import json
import sys

OWNER = sys.argv[1] if len(sys.argv) > 1 else "<OWNER>"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


cells = [
    md("# nseos - Colab driver (v2)\n\nThin front-end: every heavy step is a cached stage of the `nseos` package.\n"
       "Re-running a cell reuses finished outputs on Drive; add `--force` to recompute one stage.\n\n"
       "**Private repo:** add a GitHub token with read access to this repo as a Colab secret named `GH_TOKEN` "
       "(key icon in the left bar)."),
    code("from google.colab import drive, userdata\nimport os\ndrive.mount('/content/drive')\n"
         "ROOT = '/content/drive/MyDrive/ns_eos_project'   # artefacts persist here\n"
         "os.environ['NSEOS_ROOT'] = ROOT\ntoken = userdata.get('GH_TOKEN')\n"
         "if not os.path.exists('/content/ns-eos-amortized'):\n"
         f"    !git clone -q https://{{token}}@github.com/{OWNER}/ns-eos-amortized.git /content/ns-eos-amortized\n"
         "else:\n    !git -C /content/ns-eos-amortized pull -q\n"
         "%cd /content/ns-eos-amortized\n!pip install -q -e ."),
    code("!python -m pytest tests -q   # likelihood == simulator marginal; simulator properties"),
    md("## Core pipeline"),
    code("!python -m nseos.cli simulate"),
    code("!python -m nseos.cli train"),
    code("!python -m nseos.cli evaluate"),
    md("## Real data and nested-sampling reference"),
    code("!python -m nseos.cli grid realdata reference-real compare-real"),
    md("## Reviewer analyses (long; each is resumable)"),
    code("!python -m nseos.cli identifiability zero-noise"),
    code("!python -m nseos.cli ablate-arch   # 3 aggregators x 3 seeds; reduce with --set ablation.seeds=[0]"),
    code("!python -m nseos.cli ablate-messenger"),
    code("!python -m nseos.cli reference-synth   # CPU-heavy: dynesty on 24 synthetic catalogs"),
    code("!python -m nseos.cli importance convergence   # exact IS correction; TOV convergence"),
    code("!python -m nseos.cli economics systematics summary"),
    md("## Results"),
    code("from IPython.display import Image, Markdown, display\nfrom nseos.config import Paths, load_config\nfrom nseos.provenance import FIGURE_STAGE\nfrom pathlib import Path\npaths = Paths(load_config())\n"
         "display(Markdown(Path(paths.result('paper_numbers', 'md')).read_text()))\n"
         "for name in FIGURE_STAGE:\n    f = Path(paths.figure(name) + '.png')\n    if f.exists(): display(Image(str(f), width=900))"),
    code("# Optional: release the GPU when finished\n# from google.colab import runtime; runtime.unassign()"),
]
for c in cells:
    c["source"] = c["source"].splitlines(keepends=True)
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
with open("notebooks/colab_driver.ipynb", "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1)
print("wrote notebooks/colab_driver.ipynb for owner", OWNER)
