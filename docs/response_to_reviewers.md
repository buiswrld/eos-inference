# Response to reviewers (draft)

Reviewers: **DVmA**, **GBfw**, **ZRfB** (first venue); **HFbp**, **XKKK**, **VWKG** (ML4PS).

Text in ⟦brackets⟧ is filled from `results/*_<stage fingerprint>.json` after running the pipeline.
`results/paper_numbers_<report fingerprint>.md` collects the headline values. Do not send this until every bracket is replaced.

---

## 0. A correction we owe the reviewers

While addressing the comments we found errors in the v1 pipeline that affect submitted numbers. The full list is in `docs/CHANGES_v2.md`. The two that matter most:

1. **The Dynesty baseline used a different likelihood than the simulator the network was trained on.** It assumed equal masses, ignored the NICER mass uncertainty and used a heuristic pulsar penalty. So the v1 comparison was not apples to apples, which is exactly the concern of DVmA, VWKG (3) and HFbp. The revised likelihood is derived term by term from the simulator, and a unit test checks it against brute-force Monte Carlo of the simulator.
2. **A normalisation defect made the real catalog out of distribution.** The chirp-mass uncertainty was constant in training, and the noise level σ_Λ̃ = 0.3 Λ̃_true leaked the true value to the network. Both are fixed: noise levels are now drawn independently of θ over ranges that contain the real data, and every real feature is checked against the training distribution.

All results below are from the corrected pipeline.

---

## 1. Statistical model, likelihood and reproducibility (DVmA, VWKG-3, HFbp)

> *"exact construction of the synthetic observations, the observational noise model, the precise likelihood for the pulsar constraints … hard to verify these are solving exactly the same problem"*

We use tabulated SLy4 through 3 MeV/fm³ and a spectral core above it. At the first causal crossing the continuation saturates dP/dε = 1. If the spectral domain ends first, we hold its last dP/dε constant. The pulsar likelihood uses the bracketed maximum mass of this completed EOS. We quantify sensitivity to the continuation by conditioning the importance posterior on stars reaching their maximum inside the unextended spectral domain and reporting the retained probability and effective sample size.

The appendix now states the full generative model:
- the box prior restricted to EOSs with a bracketed nonrotating maximum mass ≥ 1.5 M☉ and a stable branch reaching 1 M☉;
- the catalog composition (N ~ U{1..50}, Dirichlet-multinomial over messengers);
- the per-messenger mass distributions;
- the noise-level distributions (independent of θ);
- the Gaussian measurement model.

It also gives the implied likelihood, including the pulsar term p(m̂|θ) = [Φ((u−m̂)/σ) − Φ((1.4−m̂)/σ)]/(u−1.4) with u = min(M_max, 2.5). Dynesty uses exactly this likelihood and the same truncated prior.

The TOV map is evaluated on a 20×20×10×5 grid. Against exact solutions at ⟦1000⟧ held-out θ, the median / 99th-percentile absolute errors are ⟦…⟧ km in R and ⟦…⟧ in log Λ. The truncation boundary is classified correctly for ⟦…⟧ % of prior draws.

Training details are in the appendix:
- data splits;
- optimiser and schedule;
- per-epoch re-simulation;
- early stopping (best epoch ⟦…⟧ of ⟦…⟧ run);
- wall time and hardware;
- package versions and seeds.

The code is released as a package with cached stages.

**Hyper-parameter selection:** ⟦state honestly, e.g. "fixed a priori; not tuned on test data"⟧.

## 2. Posterior fidelity beyond marginals (DVmA, GBfw, XKKK-3, VWKG-2, HFbp)

> *JS without units or reference; only marginals; different sample sizes; single real catalog.*

- **Units and estimator.** JS divergence is now reported in bits (log₂, bounded in [0, 1]). It uses a boundary-reflected Gaussian KDE (Scott bandwidth) on a 400-point grid within the prior bounds.
- **Matched sample sizes.** The amortised posterior is subsampled to the Kish ESS of the weighted nested samples (n = ⟦…⟧), repeated 200 times; we report the mean ± sd.
- **Noise floors.** We report both (i) JS between two independent Dynesty runs and (ii) JS between two disjoint amortised subsamples of the same size.
- **Joint posterior.** We add a classifier two-sample test (C2ST, 5-fold MLP) in 4-D: ⟦acc⟧ vs a Dynesty-vs-Dynesty floor of ⟦acc⟧. We also compare full correlation matrices (max |Δρ| = ⟦…⟧), and show an overlaid corner plot (new Fig. 1a).
- **Width.** The posterior-std ratio per parameter is ⟦…⟧. *(Replace the earlier "22 % wider γ₁" statement with these numbers.)*
- **Many catalogs.** We ran Dynesty on ⟦24⟧ synthetic test catalogs (N = 1–20, stratified), each with the same likelihood. The median JS is ⟦…⟧ vs a floor of ⟦…⟧, the median C2ST is ⟦…⟧, and the truth-in-68 % rates are ⟦…⟧ (amortised) vs ⟦…⟧ (Dynesty).
- **Derived quantities.** M_max, R₁.₄ and Λ₁.₄ are reported for both methods (GBfw): ⟦table⟧.

**Exactness on real data.** We now correct the amortised posterior by importance sampling with the exact likelihood. The proposal is the flow (density from the continuous change of variables) mixed with 10 % prior, and the target is identical to Dynesty's. On the real catalog the sampling efficiency is ⟦…⟧ (ESS ⟦…⟧ of ⟦20 000⟧). The evidence log Z = ⟦…⟧ ± ⟦…⟧ agrees with Dynesty's ⟦…⟧ ± ⟦…⟧. The headline real-data numbers are therefore asymptotically exact, and the network acts as an accelerator. Across ⟦100⟧ test catalogs the median efficiency is ⟦…⟧, a per-catalog quality diagnostic.

## 3. γ₂ and γ₃, and identifiability (all reviewers)

> *γ₂/γ₃ errors coincide with the prior-mean baseline; the zero-noise result could be an estimator failure; a Fisher/Jacobian analysis would settle it.*

We agree. We now say explicitly that the catalog-size and messenger gains concern γ₀ and γ₁ only, and that γ₂ and γ₃ remain at the prior level. Every accuracy table reports the MAE ratio to the truncated-prior median and the posterior contraction 1 − Var_post/Var_prior.

To separate the physics from the estimator we added three analyses that do not use the network:

1. **Prior width.** |Δ ln Γ(x)| ≤ max|γ_k| x^k. The γ₃ prior (±0.02) changes the adiabatic index by at most 2 %, so part of the "degeneracy" is simply a narrow prior (GBfw's point). We now say so.
2. **Fisher analysis of the TOV map.** The observables are R(m) and log Λ(m) at 9 masses plus M_max. Derivatives are central differences with a step-size check (relative change ⟦…⟧). For three noise scenarios we give the Gaussian-approximation posterior sd in prior-sd units: ⟦table⟧. The Fisher matrix has ⟦…⟧ eigenvalues above the prior level, so the data constrain ⟦≈2⟧ directions. Reaching half the prior sd on γ₂ would need ⟦…⟧× smaller uncertainties than the "current" scenario.
3. **Model-free check.** Gradient boosting trained on *noise-free* full curves reaches R² = ⟦…⟧ for γ₂ and ⟦…⟧ for γ₃.

The near-zero-noise network experiment was redone with 20k catalogs re-simulated each epoch and the main architecture. The v1 run used 400 catalogs and mis-normalised constant inputs, which is why γ₂ came out *worse* than the prior. In the revision γ₂ reaches a ratio of ⟦…⟧ to the prior.

We have reworded the claim accordingly: **⟦"γ₂ and γ₃ are weakly identified by macroscopic observables at realistic precision; γ₃'s range is limited mainly by its prior; the Fisher analysis indicates ≈2 constrained directions"⟧**.

## 4. Architecture contribution (DVmA, VWKG-1, HFbp)

> *Ridge on mean-pooled features is too weak; compare against DeepSets or a padded MLP with the same decoder.*

We added a controlled ablation in which only the aggregator changes: Set Transformer vs DeepSets (mean pool + log N) vs a padded-and-masked MLP. The flow-matching decoder, data, training budget (≤150 epochs, patience 20) and evaluation set are identical, over 3 seeds. Results (mean ± sd, MAE ratio to prior / contraction / val loss / coverage error): ⟦table⟧.

⟦If the gap is small, say so: "the amortisation, not the specific aggregator, drives the speed-up; the Set Transformer gives ⟦x⟧ % lower γ₁ error at large N."⟧

## 5. Speed-up and practical value (GBfw, VWKG-4, DVmA, XKKK)

- We now say "one aggregator pass plus a 50-step RK4 integration (200 velocity-network evaluations)" instead of "single forward pass".
- We report wall time for the same number of effective samples (⟦…⟧×) next to the raw wall-time ratio (⟦…⟧×), and the batched throughput (⟦…⟧ catalogs/s).
- **Offline cost.** Simulation took ⟦…⟧ h on ⟦12⟧ cores and training ⟦…⟧ h on an A100. The reference sampler needs the TOV grid once (⟦…⟧ min). The break-even point is ⟦…⟧ posterior analyses.
- **Concrete workflow.** We ran all ⟦127⟧ non-empty subsets of the real catalog in ⟦…⟧ s (Dynesty-equivalent ≈ ⟦…⟧ h) and report leave-one-out shifts in R₁.₄. This is the kind of influence / consistency analysis that is impractical with per-catalog sampling.
- **Scope of reuse.** Changing the prior, noise model or simulator requires re-simulation and retraining. This is stated in the Discussion.

**Prior truncation.** The training prior previously required M_max ≥ 2 M☉, which pre-empted the radio-timing observations. The cut is now 1.5 M☉ (the simulator's technical minimum), so the heavy pulsars constrain M_max through the likelihood.

**Forward-model accuracy.** Compared with a 4× denser, 100× tighter TOV solve, the production forward model's 95th-percentile errors are ⟦…⟧ km in R (⟦…⟧ of the smallest training σ_R), ⟦…⟧ relative in Λ, and ⟦…⟧ M☉ in M_max.

## 6. Training convergence and N range (XKKK-4, XKKK-5)

- The best epoch, epochs run and loss curves are reported for every run. The headline model is a 3-seed ensemble; seed-to-seed variability of every accuracy metric is reported (⟦table⟧).
- Calibration is reported for the identifiable parameters (γ₀, γ₁: coverage error ⟦…⟧) and in observable space (M_max, R₁.₄, Λ₁.₄: ⟦…⟧). γ₂/γ₃ are flagged as prior-dominated, because their calibration is trivially satisfied.
- Training now covers N = 1–50, so the N = 50 evaluation is no longer extrapolation. A nested-subset experiment for fixed θ isolates the effect of N.

## 7. Real-data realism and shared information (HFbp, VWKG, ZRfB)

- The J1614−2230 timing observation is 1.908 ± 0.016 M☉. [Arzoumanian et al. 2018, Table 8](https://arxiv.org/pdf/1801.01837)
- **Shared information.** J0740+6620's radio mass is the prior of its NICER posterior, so it is no longer counted twice. A sensitivity run with the duplicate is reported: ⟦Δ⟧.
- **Gaussian summaries.** GW and NICER constraints remain Gaussian summaries, and this is stated as a limitation. GW posteriors are now parsed from the public LIGO releases where possible, and the source used is printed.
- **Single real catalog.** This is addressed by the synthetic benchmark in § 2.

## 8. Clarity

- SBI is defined at first use.
- Figure 1 is replaced. Panel (a) is the overlaid corner plot. Panel (b) shows M–R 90 % bands for both methods with NICER 1σ/2σ ellipses and pulsar bands. Panel (c) shows M–Λ bands with the GW Λ̃ constraints at the equal-mass component mass. The caption explains every element.
- New figures: true vs predicted with 68 % bars coloured by N; SBC / coverage / TARP.

## 9. Systematics

- The crust experiment previously did not change the crust; it now does. Results: ⟦…⟧.
- We test noise covariate shift (σ × 1.5) and misreported noise (true noise 1.3 × reported σ). Coverage error for each: ⟦…⟧.
