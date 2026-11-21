# ML4PS 2026 camera-ready: notes

Main file: `ml4ps_2026_camera_ready.tex` (needs `neurips_2026.sty` and `figures/`). Upload the whole
folder to Overleaf as a new project, or replace the files of the existing one.

## Before submitting

1. **Authors.** Supply the corresponding author’s email if it should appear on the title page.
2. **Acknowledgments.** Confirm funding and support with the authors.
3. **Disclosure checklist.** Add the authors’ checklist when supplied.
4. **Co-authors.** All results and claims changed; co-authors should approve the revision.
5. **Page limit.** Confirm the final-version limit in the acceptance instructions and check the PDF again after adding the authors and disclosure.

## How each review comment is addressed

| reviewer | comment | where |
|---|---|---|
| HFbp, VWKG | MAE is not enough; need SBC, coverage, contraction, comparisons with numerical posteriors on many catalogs | Results: calibration on 15k catalogs (Fig. 2a, SBC, TARP, observable-space coverage); dynesty on 24 synthetic catalogs |
| HFbp, VWKG | Ridge baseline too weak; controlled aggregator comparison | Table 2 left: Set Transformer vs DeepSets vs padded MLP, same decoder/data/budget, 3 seeds |
| HFbp, VWKG | 4,543× / speed-up significance; offline cost; concrete workflow | Cost and use case: 139× (774×/ESS), 3× exact mode, offline costs, break-even ~570, 127-subset workflow |
| VWKG | do the network and dynesty target the same posterior? likelihood details, grid accuracy, JS method and finite-sample effects | Method: exact simulator-marginal likelihood (Monte-Carlo tested), validated grid; JS on matched n with floors; C2ST on the joint |
| VWKG, XKKK | joint posterior, not just marginals | Fig. 1 top (γ0–γ1 contours), C2ST, correlation agreement |
| XKKK | JS with unequal sample sizes | matched n = 1,802, 200 repeats, floors reported |
| HFbp | shared observational information | J0740+6620 timing mass not double-counted (dedup); duplicate variant reported |
| HFbp, XKKK, VWKG | γ2/γ3 unidentified; zero-noise result could be the estimator | Identifiability paragraph + Table 2 right: zero-noise flow, model-free regressor, Fisher, physical sensitivity |
| XKKK | compare R1.4 / Λ1.4 with literature | Results: Miller et al. 2021, LVC 2018 |
| XKKK | 80-epoch ablations, convergence, best epoch | ablations 150 epochs equal budget (stated as budget-limited); main model best epochs 367/396/115 |
| XKKK | trained to N=30, evaluated at N=50 | training now N ~ U{1..50} |
| HFbp | Figure 1 uninformative | new Fig. 1: amortized vs dynesty in parameter and observable space; Fig. 2: calibration and scaling |
| HFbp | reproducibility: features, likelihood, seeds, hyperparameters | Method section gives simulator/noise model, inputs, likelihood; 3-seed spread; hyperparameters fixed a priori |

Figures are regenerated from a finished run with `python paper/make_paper_figures.py --root ../run_v3 --out paper/ml4ps_camera_ready/figures`.
