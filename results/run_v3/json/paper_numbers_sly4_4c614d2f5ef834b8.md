# Numbers for the paper (auto-generated)

## Test split (main model)
| param | MAE | MAE prior | ratio | contraction |
|---|---|---|---|---|
| gamma_0 | 0.1093 | 0.3413 | 0.320 | 0.875 |
| gamma_1 | 0.5527 | 0.7651 | 0.722 | 0.401 |
| gamma_2 | 0.2954 | 0.3005 | 0.983 | 0.044 |
| gamma_3 | 0.0100 | 0.0100 | 1.001 | 0.013 |

Coverage error per param: [0.0019, 0.0029, 0.0075, 0.0033]
TARP max |ECP - alpha|: 0.029

Training: best epoch 367 of 400 run (max 400, patience 40), 2.39 h on NVIDIA GeForce RTX 5070, 7679748 parameters.

## Real catalog vs dynesty (same likelihood)
JS [bits], matched n=1802: [0.0014, 0.0011, 0.0008, 0.0008] (SBI floor [0.0017, 0.0012, 0.0008, 0.0008], dynesty floor [0.0006, 0.0003, 0.0003, 0.0003])
C2ST: 0.490288950531669 (floor 0.49722761596548004)
std ratio: [1.035, 1.027, 1.015, 0.993]
speed-up (wall): 139x; per effective sample: 774x
R1.4 SBI {'n_valid': 2000, 'median': 12.218515882623308, 'ci90': [11.461955505356938, 12.936490309367434], 'ci68': [11.765653238968694, 12.653157835407246]} | dynesty {'n_valid': 2000, 'median': 12.192993821304032, 'ci90': [11.478468225350298, 12.9461502357161], 'ci68': [11.75768829554379, 12.670131026648475]}

## Observable-space calibration (posterior-predictive EOS -> TOV)
| quantity | MAE | MAE prior | 68% coverage | coverage error |
|---|---|---|---|---|
| Mmax | 0.129 | 0.471 | 0.660 | 0.032 |
| R14 | 0.163 | 0.927 | 0.710 | 0.022 |
| L14 | 51.288 | 271.549 | 0.680 | 0.016 |

Coverage error, identifiable params (g0, g1): 0.002398639455782325
Prior-dominated parameters (contraction < 0.2): ['gamma_2', 'gamma_3']
Ensemble members: 3; seed variability (mean, sd): {'gamma_0': {'mae_ratio_to_prior': [0.32537808400809737, 0.0025566779359899954], 'contraction': [0.8766556724566681, 0.0024319137178491677]}, 'gamma_1': {'mae_ratio_to_prior': [0.7229575898187347, 0.0031862509201359725], 'contraction': [0.40380616572529543, 0.01220514105348002]}, 'gamma_2': {'mae_ratio_to_prior': [0.9835033684372165, 0.0024922110872478264], 'contraction': [0.04510088998691175, 0.00842461845732532]}, 'gamma_3': {'mae_ratio_to_prior': [1.00152613985062, 0.00016969443925757608], 'contraction': [0.013611378922176046, 0.006176427378935044]}}

## Importance-sampling correction (exact likelihood, real catalog)
efficiency 0.862 (ESS 17242); box-prior log Z -19.066 ± 0.003 vs dynesty -19.009 ± 0.065
Accepted-prior log Z -18.618 ± 0.004 (grid acceptance 0.6389)
Strict-support sensitivity: {'strict_posterior_mass': 0.05201658186212804, 'unresolved_posterior_mass': 0.0, 'ess': 1202.5019968909469, 'parent_ess': 17241.914597314204, 'n_positive': 18762, 'n_strict': 1417, 'n_unique_solved': 18762, 'status': 'ok', 'conditional_quantiles_05_50_95': {'gamma0': [1.1914801597595215, 1.4057050943374634, 1.6635032892227173], 'gamma1': [-1.3100452978613788, -0.538597583770752, 0.11842125654220581], 'gamma2': [-0.5647639632225037, -0.18381428747292144, 0.488552063703537], 'gamma3': [-0.018107034265995026, 9.270094597013667e-05, 0.017882967367768288], 'M_TOV': [2.0723294446665363, 2.269657881711978, 2.519831113962852], 'R14': [12.147810246906255, 12.673488277029207, 13.161107399613357], 'L14': [387.69654983027834, 533.989938120047, 713.5750906449572]}}
derived (IS-corrected): {'frac_outside_training_support': 0.0, 'frac_Mmax_below_2': 0.0, 'Mmax': {'n_valid': 2000, 'median': 2.531182997895443, 'ci90': [2.2317460795846316, 2.85568509088576], 'ci68': [2.357654415803215, 2.725014141073136]}, 'R14': {'n_valid': 2000, 'median': 12.181509675982138, 'ci90': [11.416483784056052, 12.877161380325893], 'ci68': [11.74806614193709, 12.6105564071188]}, 'L14': {'n_valid': 2000, 'median': 408.5722235614628, 'ci90': [256.51946351686723, 615.6135280446931], 'ci68': [312.72426670935783, 527.3079531222468]}}
test-catalog IS efficiency: {'median': 0.819452045943948, 'p10': 0.6220274835858244, 'frac_below_1pct': 0.0}
synthetic catalogs: {'median_abs_delta_log_z': 0.04688259303772213, 'median_efficiency': 0.8166131273648619}

## Forward-model convergence (production vs 4x denser / 100x tighter TOV)
max|dR| p95 0.0108 km (0.036 of the smallest sigma_R); max rel dLambda p95 0.0108; |dMmax| p95 0.0000
