# Changes in v2

## Current forward model and cache keys

The default is the SLy4 low-density table matched to a spectral core at 3 MeV/fm³. At the first causal crossing, dP/dε stays at one. Spectra reaching the table ceiling continue at their terminal sound speed. The stellar sequence must contain a two-sided mass peak before an EOS can enter the dataset. The technical prior also requires M_TOV ≥ 1.5 M☉ and branch coverage down to 1 M☉.

Real-catalog importance outputs include a strict-support sensitivity with posterior mass and ESS. GW integration uses 96 chirp-mass nodes per panel and 32 mass-ratio nodes. The regression suite checks all 36 combinations of masses and noise scales against a 192 × 192 reference. Observable calibration records the IDs of catalogs with insufficient draws and its effective denominators.

Artifacts carry a run label and an input fingerprint. Simulation depends on the EOS, table, prior and simulation settings. Grid construction has its own key. Training depends on the dataset and training recipe. Nested sampling includes the actual numerical catalog, grid, likelihood and sampler settings. Figure filenames follow their input stage. Use the Colab driver or `Paths` to resolve the current filenames. Earlier filenames and convergence measurements below document historical runs.

## Bugs found while addressing the reviews

The paired preproduction comparison used 320 indexed box-prior draws, seed 42, stream 1, 30 initial central pressures and 12 workers. The old endpoint rule accepted 183 EOSs, including 150 without a bracketed peak. Causal continuation accepted 203, including 169 whose maximum-mass star used the continuation. The strict rule accepted 34. Independent strict solves agreed with the primary model’s strict flag on all 320 draws.

Among the 183 draws accepted under both the old and extended rules, the median increase from the old mass limit to M_TOV was 0.561 M☉, with a 95th-percentile increase of 1.834 M☉. R₁.₄ and Λ₁.₄ were identical at the reported precision on this paired subset. Thirteen draws reached their peak above the 1,500 MeV/fm³ table edge without a prior causal crossing. All thirteen failed the 1.5 M☉ acceptance cut. This comparison describes the prior.

These affect the v1 / submitted numbers.

| # | v1 problem | Consequence | v2 fix (module) |
|---|---|---|---|
| **B1** | The GW chirp-mass σ was the constant `0.01` in every training catalog, so its normalisation sd was clipped to `1e-6`. GW170817 has σ_𝓜 = 0.001, which gave the encoder an input of (0.001−0.01)/1e-6 ≈ **−9000**. | The real-data input was far out of distribution. In the v1 notebook run, the amortised real-data posterior had **M_max = 1.76 [1.49, 1.99] M☉**, entirely below the training prior's M_max ≥ 2, and disagreed with Dynesty (mean JS 0.23, γ₀ 0.74 vs 1.00). | σ's are drawn log-uniformly over ranges covering the real data. Encoders see standardised **log σ** (`simulator.py`, `model.py`), and real features are checked against the training distribution (`realdata.ood_report`). |
| **B2** | σ_Λ̃ = max(50, 0.3·Λ̃_true) was an input to the network. | σ revealed Λ̃_true almost noise-free, so the synthetic accuracy was optimistic. The real GW170817 (σ = 190 ≠ 0.3·300) was off the training manifold. | σ_Λ̃ ~ LogU(40, 800), independent of θ (`simulator.py`). Tested in `tests/test_simulator.py`. |
| **B3** | The Dynesty likelihood ≠ the simulator. GW assumed equal masses from 𝓜 and ignored 𝓜 noise. NICER treated M_obs as the true mass. The pulsar term was a one-sided penalty instead of m ~ U(1.4, min(M_max, 2.5)) + noise. The prior was a plain box (the simulator truncates at M_max ≥ 2). The grid was a coarse 10⁴ with NaN holes neighbour-filled and never validated. | The two methods **targeted different posteriors**, which was the central reviewer concern. | Every likelihood term is the exact marginal of the simulator (quadrature over nuisance masses, closed-form pulsar term), with the same truncated prior, a finer 20×20×10×5 grid and validation of interpolation error and truncation boundary (`likelihood.py`). Tested against brute-force Monte Carlo (`tests/test_likelihood_matches_simulator.py`). |
| **B4** | The crust experiment computed `K_crust_varied` and never used it. | "Insensitive to crust" tested nothing. | `gamma_crust` is threaded through `build_eos` / `compute_mr_curve` (`physics.py`, `stages.stage_systematics`). |
| **B5** | The zero-noise model had 400 training catalogs and copied the main model's normalisation, so constant σ=1e-4 inputs became ≈ −10⁴ sd. Its prior baseline was the box centre. | γ₂ was *worse than the prior mean*, which is an optimisation failure, not evidence of non-identifiability. | 20k catalogs re-simulated each epoch, its own normalisation, the same architecture as the main model, and the truncated-prior baseline. Plus a model-free regressor and a Fisher analysis (`stages.stage_zero_noise`, `identifiability.py`). |
| **B6** | Catalog-size tests used the first N observations, which were always GW first. Training N ≤ ~32, yet N=50 was claimed. | Size was confounded with messenger type, and N=50 was extrapolation. | N ~ U{1..50}, Dirichlet-multinomial composition, shuffled order. Evaluation binned by N, plus nested subsets for fixed θ. |
| **B7** | Overfitting: val loss was best at epoch 108 of 400 (train 0.64 vs val 1.49 at the end). This was undocumented. | | Per-epoch re-simulation from stored curves, patience-based early stopping, fixed-RNG validation, and logging of best/total epochs and wall time (`train.py`). |
| **B8** | J0740+6620's mass was used twice (radio + the NICER mass prior). | Shared information was double counted. | `policy="dedup"` by default; `duplicate` is kept as a sensitivity run. |
| **B9** | The HDF5 parser failed on both files (GWTC-1 layout; the GW190425 URL was a 404 page), and the literature fallback was used silently. | | Generic parser (component masses → source-frame 𝓜, Λ̃), the correct LIGO-P2000026 URL, and status reporting (`realdata.py`). |
| B10 | Posterior samples outside the prior box were *clipped* onto the boundary. | Spikes at the box edges distorted marginal comparisons. | Rejection sampling inside the box. The out-of-box fraction is reported (`inference.py`). |
| B11 | R₁.₄ / Λ₁.₄ came from the nearest of 30 central pressures. | | Interpolated on the stable branch. Samples outside the training support are counted. |
| **B12** | The prior was truncated at M_max ≥ 2.0 M☉. | The prior already encoded what the heavy pulsars say, so the pulsar data, the pulsar ablation and the real-data M_max posterior were partly circular. It also rejected ~75 % of prior draws. | The cut is lowered to the simulator's technical minimum (M_max ≥ 1.5 M☉), so the radio-timing observations constrain M_max through the likelihood (`config.sim.MMAX_MIN`). |
| **B14** | For many EOS the M–R branch never turns over: it ends where the EOS becomes acausal at the centre. "M_max" was then the mass at the last of 30 coarse central pressures. | M_max was biased **low** by up to 0.19 M☉ (95th percentile 0.16 M☉ vs a 4× denser, tighter solve, on 32 EOS). This entered the pulsar likelihood, the prior cut and the real-data M_max posterior. The bias was resolution-dependent and present in v1. | Bisection on central pressure between the last valid and first rejected star, plus golden-section refinement of interior maxima. 95th-percentile error is now 0.0005 M☉, for +22 % TOV cost. Found by the new `convergence` stage. |
| B15 | Λ(M) was interpolated linearly between the 30 branch points, but Λ ∝ M⁻⁶. | 95th-percentile Λ error was 6.6 %, comparable to the most precise synthetic GW events. | Log-space interpolation through one shared helper (`simulator.interp_lambda`) used by the simulator, likelihood and analyses. Error is now 1.6 %. |
| **B13** | The TOV integration was abandoned after 0.5 s of *wall-clock* time. | Acceptance, and therefore the effective prior, depended on machine load and parallelism. | A deterministic limit on RHS evaluations. It reproduces accepted curves to 10⁻⁶ km (`physics.py`). |

## New analyses (reviewer requests)

| Request | Stage / output |
|---|---|
| Exact simulator, noise model and pulsar likelihood specification | `simulator.py` docstring, `likelihood.py` docstring |
| Statement that γ₂/γ₃ are at the prior level; per-parameter accuracy vs prior | `evaluate` → `eval_main.accuracy` |
| True-vs-predicted plot with 68 % bars coloured by N | `figures/true_vs_pred_main_v2` |
| Controlled architecture ablation, multiple seeds | `ablate-arch` → `ablation_arch_v2.json` |
| SBC rank histograms, coverage curves, TARP, contraction | `figures/calibration_main_v2` |
| Identifiability independent of the estimator | `identifiability` → Fisher/CRB, eigen-spectrum, scans; `zero-noise` → model-free GBR |
| JS in bits, KDE spec, matched n, sampler-noise floor, joint (C2ST, correlations), derived quantities for Dynesty | `compare-real` |
| Overlaid corner plot; better Fig. 1 (bands, GW constraints) | `figures/real_corner_overlay_v2`, `figures/fig1_mr_mlambda_v2` |
| Reference posteriors on many synthetic catalogs | `reference-synth` |
| Offline costs, break-even, concrete workflow | `economics` |
| Seeds, best epoch, compute, versions | `runs/*/train_info.json`, `summary` |

## Additional rigor and methodology (beyond the reviews)

| Addition | Why | Where |
|---|---|---|
| **Importance-sampling correction** with the exact likelihood. The defensive mixture proposal is 90 % flow + 10 % prior box, using the flow's exact log-density from the continuous change of variables. | Makes the real-data posterior asymptotically exact, with the network acting only as an accelerator. Gives the evidence log Z, directly comparable to dynesty, and a per-catalog quality diagnostic (efficiency = ESS/n). | `importance.py`, stage `importance`; log-density verified in `tests/test_flow_logprob.py` |
| **3-seed ensemble** for the headline model | Seed variance is reported, and the posterior approximation is better. | `train.ensemble_seeds`, `model.EnsembleModel` |
| **Forward-model convergence test.** The production TOV (30 central pressures, rtol 10⁻⁶) is compared with a 4× denser, 100× tighter solve. | Discretisation error is bounded against the smallest training σ's. | stage `convergence` |
| **Observable-space calibration** (M_max, R₁.₄, Λ₁.₄). Headline calibration uses γ₀/γ₁, and γ₂/γ₃ are flagged as prior-dominated. | A prior-dominated posterior is trivially calibrated, so averaging over all four γ's flattered the method. | `evaluate` (`derived_test`, `coverage_error_identifiable_g0_g1`) |
| Early stop of each M–R curve past M_max; 100k EOS by default | Cheaper simulation. Early stop measured at only ~2 %; most savings come from the higher acceptance after B12. | `physics.compute_mr_curve`, `config.data` |
