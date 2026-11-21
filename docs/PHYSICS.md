# Stellar model and mass limit

[← back to README](../README.md)

## Equation of state

The low-density EOS uses the LALSuite SLy4 table (`nseos/sly4.dat`) through a pressure of 3 MeV/fm³, including the outer core. The four spectral coefficients γ₀…γ₃ describe the core above this match:

Γ(x) = exp(γ₀ + γ₁ x + γ₂ x² + γ₃ x³), x = log(P / P_match) / log(P_max / P_match) ∈ [0, 1]

(see `physics.spectral_gamma` for the exact constants).

At the first causal crossing (c_s = c) we continue with ε(P) = ε★ + (P − P★), with pressure and energy density in the same units. This selects the stiffest causal continuation. A spectrum that stays causal to 1,500 MeV/fm³ continues at its terminal dP/dε.

## Stellar structure

Each EOS is integrated through the TOV equations with the tidal (Love-number) equation. The tidal equation uses the corrected ν′ factor of 2, and a small-compactness series for the k₂ denominator. The integration has a deterministic limit on right-hand-side evaluations, so acceptance does not depend on machine load.

## Maximum mass

The pulsar mass limit is the bracketed peak of the nonrotating mass sequence. The scan extends to 240,000 MeV/fm³ if needed. A solver failure or an unbracketed peak at the cap produces a rejection reason. Dataset records contain the matching pressure, boundary type, peak pressure and strict-support flag.

## Strict-causality sensitivity

The strict sensitivity in the real-catalog importance stage conditions the weighted posterior on an exact stellar solve whose peak is bracketed within the original spectral domain. Its outputs include retained posterior mass, conditional ESS and normalized sample weights.

The strict probability is the sum of the parent importance weights on classified strict samples. Conditional weights are divided by this probability. The parent posterior uses the interpolated grid likelihood, so the sensitivity also inherits its interpolation error. Unresolved numerical membership is reported separately.

Reproduce the three-policy comparison with

```bash
python tools/compare_causality.py --draws 320 --workers 12 --output causality-comparison.json
```

Each policy uses the same indexed box-prior draws. The historical endpoint in the lenient arm can be a cutoff mass. Only the extended and strict arms require a bracketed maximum.

## Prior

Box prior γ₀ ∈ [0.2, 2.0], γ₁ ∈ [−1.6, 1.7], γ₂ ∈ [−0.6, 0.6], γ₃ ∈ [−0.02, 0.02], truncated to EOSs with M_TOV ≥ 1.5 M☉ and a stable branch down to 1 M☉ (the simulator's technical minimum; about 64 % of box draws pass).

## Observation model

See the docstrings of `nseos/simulator.py` (generative model) and `nseos/likelihood.py` (its exact marginal likelihood):

- **GW:** component masses drawn uniformly, observed as (Λ̃, chirp mass) with log-uniform noise levels independent of θ.
- **NICER:** (M, R) with independent log-uniform noise levels.
- **Radio / optical pulsar masses:** m ~ U(1.4, min(M_TOV, 2.5)) plus Gaussian noise.
- **Catalog size and mix:** N ~ U{1..50}, with messenger composition drawn from a Dirichlet-multinomial and shuffled order.
