# Paired comparison of neutron-star maximum-mass rules

The causal extension accepted 203 of 320 production-prior draws, all with bracketed maximum masses. The other 117 had maximum masses below the prior threshold. The strict rule accepted 34 EOSs. The 34 were the same EOSs identified by the strict flag on the extended solves.

## Acceptance and cost

| Rule | Accepted / 320 | Acceptance | Accepted without a bracketed peak | Wall seconds per accepted EOS |
|---|---:|---:|---:|---:|
| Previous cutoff | 183 | 57.2% | 150 | 0.422 |
| Causal extension | 203 | 63.4% | 0 | 0.614 |
| Strict | 34 | 10.6% | 0 | 2.368 |

The cost divides elapsed wall time for all 320 attempts by the number accepted, using 12 worker processes. At the measured extension throughput, 100,000 accepted EOSs would take approximately 17.1 hours for the stellar solves. Production catalog generation, disk writes, training and inference add time. The process pool was created once, with the previous cutoff policy running first.

## Paired shifts

Differences below subtract the previous cutoff result from the extended result on the 183 EOSs accepted under both rules.

| Quantity | 5th percentile | Median | 95th percentile | Largest absolute shift |
|---|---:|---:|---:|---:|
| Upper mass limit, M☉ | 0.000 | +0.561 | +1.834 | 2.015 |
| Radius at 1.4 M☉, km | 0 | 0 | 0 | 0 |
| Tidal deformability at 1.4 M☉ | 0 | 0 | 0 | 0 |

For 169 of 203 accepted extended EOSs, the maximum-mass star reached the continuation. Twenty additional EOSs passed the mass cut under the extension. The unchanged canonical-star results apply to the paired 183. Across the 34 EOSs accepted by both the strict and extended rules, all three numerical observables agreed at the stored precision. The strict fraction of this accepted **prior** sample was 16.7%. The observational posterior fraction is estimated separately with importance weights.

Thirteen of the 320 extended EOSs had a peak above the spectral table ceiling without crossing the causal limit inside the table. All thirteen failed the 1.5 M☉ cut. There were no extended-arm rejections for reaching the hard pressure cap without a turnover.

## Reproduction

The run used seed 42, simulation stream 1 and production draw indices 0–319. All three policies received identical coefficients on each index. The initial central-pressure scan used 30 points. EOSs remained subject to the 1.5 M☉ mass cut and branch coverage down to 1 M☉.

The primary model matches SLy4 to the spectral core at 3 MeV/fm³. At the first causal crossing it continues at dP/dε = 1. A spectrum that stays causal to 1,500 MeV/fm³ continues at its last dP/dε. The extended search ceiling is 240,000 MeV/fm³. The strict rule requires a peak bracketed inside the original spectral domain. The previous cutoff rule permits a final mass on a still-rising sequence.

```shell
python tools/compare_causality.py --draws 320 --workers 12 --output causality-comparison.json
```

The adjacent JSON contains each draw's coefficients, policy, matching and peak diagnostics, acceptance decision, rejection reason, observables and measured duration.
