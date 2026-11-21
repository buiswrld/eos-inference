# Paper text changes (checklist)

## Abstract / introduction
- [ ] Define **SBI** (simulation-based inference) at first use, along with NPE, SBC and TOV.
- [ ] Replace "single forward pass" with "one aggregator pass and a 50-step RK4 integration of the learned flow (200 network evaluations)".
- [ ] Replace the speed-up number with the v2 value and qualify it: "online, amortised; excludes ⟦h⟧ of simulation and ⟦h⟧ of training".

## Method
- [ ] For the J1614−2230 catalog entry, cite Arzoumanian et al. (2018), NANOGrav 11-year data set, Table 8, for 1.908 ± 0.016 M☉. Demorest et al. (2010) describes the earlier measurement. [Source table](https://arxiv.org/pdf/1801.01837)
- [ ] Describe the SLy4 match at 3 MeV/fm³, the causal continuation and terminal-sound-speed rule, and the two-sided bracket defining M_TOV. Give retained posterior mass and conditional ESS for the strict sensitivity.
- [ ] Add an appendix "Statistical model", copying the generative model from `nseos/simulator.py` and the likelihood from `nseos/likelihood.py`.
- [ ] State that the encoders take log σ and that σ's are drawn independently of θ.
- [ ] State per-epoch re-simulation, early stopping and the best epoch.
- [ ] State that N ∈ [1, 50] is used for both training and evaluation.

## Results
- [ ] Every accuracy table: add the MAE ratio to the truncated-prior median and the contraction. Say explicitly that γ₂ and γ₃ stay at the prior level and that the N-dependence and messenger gains concern γ₀ and γ₁.
- [ ] Replace Ridge as the main baseline with the controlled aggregator ablation table (keep Ridge as a sanity row).
- [ ] Dynesty comparison: JS **in bits**, matched n, both floors, C2ST, correlation matrices, derived quantities for both methods, and the synthetic-catalog benchmark.
- [ ] Identifiability: the prior-width remark, the Fisher table, the model-free R², and the revised zero-noise result. Soften "structural identifiability limit" to the wording in response §3.
- [ ] Real-data result: dedup policy, with the sensitivity to the duplicate.

- [ ] New subsection "Exact correction by importance sampling": the flow as proposal, efficiency, log Z vs dynesty. Quote the real-data numbers from the IS-corrected posterior.
- [ ] State the training prior: box ∩ (causal, M_max ≥ 1.5 M☉, branch reaches 1 M☉). Note explicitly that the pulsars, not the prior, constrain M_max.
- [ ] Appendix: forward-model convergence numbers, the 3-seed ensemble, and seed variability.

## Figures
- [ ] Fig. 1 → overlaid corner plot (`real_corner_overlay_v2`) + M–R/M–Λ bands (`fig1_mr_mlambda_v2`). New caption: "Blue: amortised posterior; red: nested sampling with the identical likelihood and prior. Contours: 68/95 % HPD. (b) Median and 90 % band of R(M) from 300 posterior EOS; ellipses: NICER Gaussian summaries (1σ solid, 2σ dashed); orange: radio-timing masses ±1σ. (c) Λ(M) bands; squares: GW Λ̃ at m = 𝓜 2^{1/5} (equal-mass Λ̃ = Λ(m))."
- [ ] Add `true_vs_pred_main_v2` and `calibration_main_v2` (appendix if space is tight).

## Discussion / limitations
- [ ] Practical value: the all-subsets / leave-one-out workflow and the break-even number of analyses.
- [ ] Reuse scope: a new prior, noise model or simulator requires retraining.
- [ ] Gaussian summaries of GW/NICER posteriors and shared information between messengers.
- [ ] Replace the earlier crust statement (the v1 test was a no-op) with the v2 result.
