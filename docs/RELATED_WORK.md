# Related work and novelty positioning

[← back to README](../README.md)

Literature search of 2026-11-20. Some entries rest on abstracts or partial full-text reads. **Re-check each
before citing**, and follow the reference lists of the closest papers (§1) on INSPIRE / ADS for anything missed.

## 1. Closest prior work (must be cited and contrasted)

| paper | what it does | what it does *not* do (relative to this work) |
|---|---|---|
| Ferreira, Carvalho, Bejger & Providência, *Set Transformer inference of the NS EOS from stellar observations*, [arXiv:2609.31081](https://arxiv.org/abs/2609.31081) (Sep 2026) | Set Transformer on unordered sets of 5–35 stars (M–R, M–Λ or both); outputs P(n) or c_s²(n) on a density grid; calibration curves and 2σ coverage | Per-density heteroscedastic **Gaussian** output, not a joint posterior; no comparison with a likelihood-based sampler; **no real data**; does not mix GW / NICER / radio messengers; no importance sampling or evidence |
| Thakur, *Fast Bayesian updating of the NS EOS with neural posterior and evidence estimation*, [arXiv:2610.05121](https://arxiv.org/abs/2610.05121) (Oct 2026) | **Flow-matching posterior estimation + importance sampling** (A-NET+IS), Deep Sets over NICER posterior samples, evidence network, agreement with UltraNest, real data (nuclear, NICER, GW170817, pQCD); 15–240× faster per configuration | RMF couplings, not a phenomenological core EOS; conditions on **exactly three** NICER sources; GW170817 enters only through the IS likelihood and a scalar control, not as catalog input; no radio-mass catalog; **no SBC / coverage / TARP** reported |
| Carvalho, Ferreira, Bejger & Providência, *NS-UNO: NS EoS inference from an unconstrained number of observations*, [arXiv:2608.30573](https://arxiv.org/abs/2608.30573) (31 Aug 2026, **before our 20 Sep submission**) | NPE with a hierarchical **DeepSets** encoder + conditional normalizing flow over a **variable number** of M–R observations (each a set of posterior samples); posterior output | M–R only, no mixing of GW / NICER / radio messengers; no attention / Set Transformer; real data only as a qualitative demonstration (per abstract) |
| Carvalho, Ferreira, Bejger & Providência, *Neural posterior estimation of NS EOSs*, [arXiv:2507.23506](https://arxiv.org/abs/2507.23506), Phys. Rev. D ([10.1103/6396-4dcy](https://link.aps.org/doi/10.1103/6396-4dcy)) (2025) | NPE with conditional normalizing flows from masses, radii and tidal deformabilities; polytropic and GP priors; physics-informed loss; "well-calibrated" | Synthetic data only (per abstract); no same-likelihood sampler comparison stated; no mixed-messenger variable catalogs stated |
| Thakur & Malik, *Amortized SBI of RMF couplings*, [arXiv:2606.25446](https://arxiv.org/abs/2606.25446) (Jun 2026) | Amortized NPE validated against PyMultiNest, TARP calibration | Conditions on nuclear observables + M_max, not on stellar-observation catalogs; NICER / GW listed as future work; no real data |
| McGinn et al., *Rapid NS EOS inference with normalising flows* (ASTREOS), [arXiv:2403.17462](https://arxiv.org/abs/2403.17462) (2024) | Normalizing-flow EOS inference from GW data, consistent with LVC results in < 1 s | GW only |

## 2. Other relevant work

- Brandes, Whiteson et al., neural SBI from telescope spectra (neural *likelihood* estimation + HMC), [arXiv:2403.00287](https://arxiv.org/abs/2403.00287).
- Gonçalves da Silva & Farias, point estimates vs posteriors for the EOS (MLPs underestimate high-density uncertainty; flows needed), [arXiv:2608.19019](https://arxiv.org/abs/2608.19019). Supports using a generative posterior rather than a regressor.
- RoyChowdhury, Ray, Creighton & Ghosh, *Rapid EOS inference from multiprobe datasets* (GW + NICER + radio), [arXiv:2611.24664](https://arxiv.org/abs/2611.24664) (Nov 2026): R₁.₄ = 11.93 (+1.02/−0.83) km, M_max = 2.31 ± 0.23 M☉. Method details not yet checked. Useful for comparison of the physics results.
- Earlier deterministic-network EOS reconstruction: Fujimoto, Fukushima & Murase, [arXiv:2101.08156](https://arxiv.org/abs/2101.08156); Soma et al., [arXiv:2201.01756](https://arxiv.org/abs/2201.01756); Ventagli & Saltas, [arXiv:2405.17908](https://arxiv.org/abs/2405.17908).
- Methods: Dax et al., *Neural importance sampling for rapid and reliable GW inference* (DINGO-IS), [arXiv:2210.05686](https://arxiv.org/abs/2210.05686); Wildberger et al., *Flow matching for scalable SBI* (FMPE), NeurIPS 2023 ([arXiv:2305.17161](https://arxiv.org/abs/2305.17161)).
- Spectral EOS: Lindblom, [arXiv:1009.0738](https://arxiv.org/abs/1009.0738); Lindblom & Indik, [arXiv:1207.3744](https://arxiv.org/abs/1207.3744), [arXiv:1310.0803](https://arxiv.org/abs/1310.0803) (two to three coefficients usually suffice to represent known EOSs).

**Architectural overlap is partial in both cases.** Ferreira+ shares only the encoder choice (a Set Transformer
over stars) but is a heteroscedastic regressor, not a posterior estimator. Thakur shares only the output side
(FMPE + importance sampling) but conditions on a fixed set of three NICER sample clouds via Deep Sets. Neither
combines type-specific encoders over a variable mixed-messenger catalog with an exact-correctable generative
posterior.

## 3. What is *not* new (do not claim)

- Amortized / neural posterior estimation for the NS EOS (Carvalho+ 2025, McGinn+ 2024).
- Set Transformers on variable-size sets of NS observations (Ferreira+ 2026).
- Flow matching + importance sampling + nested-sampling validation for NS EOS, with evidence (Thakur 2026).
- Flow matching for SBI (Wildberger+ 2023) and neural importance sampling (Dax+ 2023).

## 4. What appears new (claim with "to our knowledge")

1. **One amortized posterior over mixed-messenger catalogs.** A single network takes a variable-size (1–50) catalog mixing GW (Λ̃, 𝓜), NICER (M, R) and radio/optical masses, with per-observation noise levels as inputs. It matches messenger-specific retrained models. None of the papers above mixes all three messenger types in one variable catalog.
2. **Same-target validation.** The reference sampler uses the *exact marginal likelihood of the simulator*, verified against brute-force Monte Carlo, so network and nested sampling provably target the same posterior. It is indistinguishable on real data (JS ≤ 0.0014 bits, at the sampling-noise floor; C2ST 0.49) and on 24 synthetic catalogs.
3. **Calibration and exactness together.** SBC, coverage and TARP on 15,000 held-out catalogs, and observable-space calibration of M_TOV, R₁.₄ and Λ₁.₄, *plus* importance-sampling exactness (86 % efficiency on real data), with an evidence that agrees with dynesty and is 20× more precise. Thakur (2026) has IS and evidence but no calibration suite. Ferreira+ (2026) has calibration but no sampler comparison or real data.
4. **Estimator-independent identifiability of the spectral coefficients.** A Fisher analysis, a near-zero-noise model and a model-free regressor all show γ₂ and γ₃ are unconstrained even at zero noise. This turns Lindblom's "two to three coefficients suffice" into a measured statement about what the data can recover.
5. **Workflow and honesty results.** All 127 sub-catalogs analysed in 24 s (vs ~4.1 h of dynesty), leave-one-out influence, messenger and architecture ablations, quantified crust dependence, and a corrected speed-up (139× fast / ~3× exact; break-even ~570 / ~930 analyses).

## 5. Suggested introduction paragraph (draft)

> Neural posterior estimation has recently been applied to the neutron-star EOS from stellar observables
> [Carvalho+ 2025; McGinn+ 2024], including permutation-invariant Set Transformers over variable-size sets of
> stars [Ferreira+ 2026] and flow-matching posteriors corrected by importance sampling [Thakur 2026]. These
> studies either condition on a fixed set of sources, treat a single messenger type, or do not compare against
> a likelihood-based sampler targeting the same posterior. Here we train a single amortized flow-matching
> posterior over catalogs of 1–50 observations that mix gravitational-wave, NICER and radio-timing
> measurements, validate it against nested sampling with the exact marginal likelihood of the simulator, and
> show that it is calibrated, can be made exact by importance sampling, and identifies which EOS parameters the
> data can and cannot constrain.
