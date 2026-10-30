# DIC data report (measured, scripted)

Generated 2026-10-30 13:02:06. Every number below is computed by VFM Studio from the raw files; thresholds only decide what is listed, not whether a test is acceptable.

## Overview

| Test | Steps / facets | Dominant Δκ at final step [1/mm] | Frame check | Interior noise ÷ model (κxx/κyy/κxy) | Inferred curvature filter [facets] (mismatch) | Subset ≈ [px] | Flagged zones | Automatic checks |
|---|---|---|---|---|---|---|---|---|
| Specimen_A1_PB | 11 / 7,097 | κyy +0.00892 | consistent | 1.07 / 1.29 / 1.05 | 15 (0.273) | 25 | 0 | 3 |

## Dataset and format check

- **Files:** 11 raw Vic-3D exports in 1 test(s); all match the expected 33-column format
- **Cross-file checks:** same facet step in every file of a test; every file shares at least half its facets with the first
- **Conversion:** κ = (−2·cxx, −2·cyy, +2·cxy); κxy is the engineering twist; ε = (exx, eyy, 2·exy)
- **Increments:** Δ quantities are relative to step 1 of each test, in the order the user confirmed

## Specimen_A1_PB

### Specimen and test

- **Folder:** VFM Studio / DemoData / Specimen_A1_PB
- **Measured region:** 9.45 × 32.9 mm (295 mm²); facet pitch 0.207 × 0.201 mm (7 px step); camera image x → +X, image y → −Y
- **Internal holes:** 0 (cut-outs open to an edge are not counted)
- **Width along Y (bottom→top) [mm]:** 8.68, 8.68, 8.89, 8.89, 8.89, 9.1, 9.1, 9.1, 9.3, 9.3, 9.1, 6.2
- **Steps:** 11

| Machine channel | Step 1 | Last step | Min | Max |
|---|---|---|---|---|
| Tx | -0.4 | 0.02 | -0.4 | 0.522 |
| Ty | 0.0551 | 37.5 | 0.0551 | 37.5 |
| Tz | 2.72 | 2.73 | -1.18 | 4.85 |
| θx | 0.00351 | 0.226 | -0.00717 | 0.226 |
| θy | -0.000749 | -20 | -20 | -0.000749 |
| θz | 0.000586 | -0.635 | -0.635 | 0.0364 |

Moments in N·mm, rotations in degrees.

### Deformation (measured)

- **Reference / final step:** 1 → 11 (step with the largest change of the dominant machine moment)
- **Dominant machine moment / rotation:** Ty / θy
- **Interior mean Δκ [1/mm]:** κxx -0.0024, κyy +0.00892, κxy -0.000859 (dominant κyy)
- **Secondary ÷ dominant:** κxx -0.269, κxy -0.0963
- **Linearity vs load (R²):** κxx 0.98, κyy 0.981, κxy 0.84
- **ΔW quadratic fit [1/mm]:** κxx -0.00229, κyy +0.0088, κxy -0.000458; tilt about X -9.48°, about Y 0.283°
- **Curvature frame check:** consistent: the Vic curvature tensor matches the ΔW surface as exported (mismatch as exported 0.0234, with xx/yy swapped 1.28)

| Band (along Y) | Width [mm] | Δκxx | Δκyy | Δκxy | ΔW [mm] |
|---|---|---|---|---|---|
| bottom | 8.89 | -0.0022 | +0.00856 | -0.000934 | -0.158 |
| lower | 8.89 | -0.00279 | +0.00912 | -0.00101 | -0.682 |
| middle | 9.1 | -0.00291 | +0.00858 | -0.000919 | -1.59 |
| upper | 9.3 | -0.0027 | +0.009 | -0.000293 | -2.87 |
| top | 9.3 | -0.0019 | +0.00943 | -0.001 | -4.5 |

### Noise vs model

Load variable machine channel Ty [N·mm]; 8 residual dof per facet (median-unbiased).

|  | Measured interior [1/mm] | Model interior [1/mm] | Interior ratio | All-facet ratio | Edge amp. meas / model | Legacy envelope ÷ measured | Map corr. | SNR final step |
|---|---|---|---|---|---|---|---|---|
| κxx | 0.000515 | 0.00048 | 1.07 | 1.27 | 3.17 / 3.07 | 2.35 | 0.8 | 3.93 |
| κyy | 0.000654 | 0.000508 | 1.29 | 1.42 | 1.61 / 1.26 | 1.45 | 0.72 | 12.4 |
| κxy | 0.00054 | 0.000515 | 1.05 | 1.24 | 2.63 / 1.91 | 2.52 | 0.76 | 1.35 |

### DIC processing parameters and curvature filter

- **Recorded in the export:** facet step 7 px; pitch 0.207 × 0.201 mm; image scale 0.0295 × 0.0287 mm/px
- **Subset size (estimated):** ≈ 25 px from the noise correlation length (half-correlation lag [facets] U along x 1.66, U along y 1.89, V along x 1.81, V along y 1.77, W along x 1.7, W along y 1.72)
- **Curvature filter (inferred):** 15 facets (3.1 × 3.01 mm), gaussian, edge weight 5% of centre, σ 0.583 mm; mismatch to the exported Δκ 0.273 (0.926 of its variance reproduced): reproduced closely
- **Other candidates (mismatch):** 13 facets gaussian, edge weight 20% of centre (0.31); 17 facets gaussian, edge weight 2% of centre (0.312); 13 facets gaussian, edge weight 10% of centre (0.339); best uniform 11 facets (0.505); noise-model stencil, uniform 15 facets (0.751)
- **Refit ÷ exported interior noise at the inferred filter:** κxx 0.951, κyy 0.968, κxy 0.931
- **Noise ∝ size^p:** measured p = -1.83, uncorrelated W noise would give -3.18
- **Smallest filter size reaching:** SNR ≥ 3: 7 facets; SNR ≥ 10: 13 facets; smoothing bias index > 1.1: none in the sweep; full-window facet fraction < 0.5: 21 facets

Curvature filter sweep (gaussian, edge weight 5% of centre; signal = interior mean Δκyy at step 11):

| Filter [facets] | Window x [mm] | Full-window facets | Interior noise κxx / κyy / κxy [1/mm] | SNR | Smoothing bias index | Mismatch to export |
|---|---|---|---|---|---|---|
| 5 | 1.03 | 0.88 | 0.00458 / 0.00481 / 0.00539 | 1.83 | 1 | 6.14 |
| 7 | 1.45 | 0.83 | 0.00282 / 0.00294 / 0.00319 | 3.01 | 1 | 3.88 |
| 9 | 1.86 | 0.77 | 0.00173 / 0.00181 / 0.00188 | 4.89 | 1.01 | 2.26 |
| 11 | 2.27 | 0.72 | 0.00108 / 0.00117 / 0.00116 | 7.6 | 1.03 | 1.18 |
| 13 | 2.69 | 0.67 | 0.00071 / 0.000818 / 0.000748 | 10.8 | 1.03 | 0.507 |
| 15 (inferred) | 3.1 | 0.61 | 0.00049 / 0.000635 / 0.000505 | 13.9 | 1.04 | 0.273 |
| 17 | 3.51 | 0.56 | 0.000364 / 0.000536 / 0.000361 | 16.4 | 1.05 | 0.451 |
| 19 | 3.93 | 0.52 | 0.000285 / 0.000483 / 0.000268 | 18.3 | 1.06 | 0.62 |
| 21 | 4.34 | 0.47 | 0.000236 / 0.000452 / 0.000209 | 19.5 | 1.07 | 0.738 |
| 23 | 4.75 | 0.42 | 0.000205 / 0.000432 / 0.000171 | 20.4 | 1.08 | 0.817 |
| 25 | 5.17 | 0.38 | 0.000184 / 0.000421 / 0.000144 | 21 | 1.08 | 0.87 |
| 27 | 5.58 | 0.33 | 0.000171 / 0.000411 / 0.000125 | 21.4 | 1.09 | 0.905 |
| 29 | 6 | 0.29 | 0.000163 / 0.000404 / 0.000112 | 21.9 | 1.09 | 0.929 |
| 31 | 6.41 | 0.25 | 0.000159 / 0.000394 / 0.000104 | 22.4 | 1.09 | 0.945 |

### Read quality

- **Facets:** 7,097 read in at least one step, 7,096 in every step
- **Vic-3D sigma median [px]:** 0.00733 at step 1 → 0.0122 at the last step (p95 0.0087 → 0.0185); σZ median 0.000731 → 0.00103 mm
- **Zone noise ÷ model:** 1.05 to 1.42 (5 bands along Y (bottom→top) × 3 along X (left→right), on the bounding box of the measured region)
- **Poorly read zones:** none crosses the dropout, correlation-sigma or noise thresholds

### Automatic checks

- **signal:** κxy signal-to-noise at the final step is 1.35
- **deformation:** κxx is -0.27× the dominant κyy
- **curvature filter:** the noise model's stencil (uniform 15 facets) reproduces the exported curvature with mismatch 0.75, against 0.27 for the inferred filter (15 facets, gaussian, edge weight 5% of centre)

## Noise model used

σ_c(x) = σ_int,c · s_c(x),  s_c = 1 + (g_c − 1)/c,  g_c = ‖w_c(x)‖/‖w_c(full)‖ (Vic-3D 15×15 local quadratic stencil, truncated at every boundary of the facet mask). Interior floor [1/mm]: κxx 0.00048, κyy 0.000508, κxy 0.000515. Calibrated on 45_deg Laminate A1_PB, A1_Twist, A2_PB, A2_Twist, B1_PB, B1_Twist, B2_PB, B2_Twist, 45_deg_optimized Test_2, Test_3 (cutout coupons). Known limits: calibrated on 45° laminate coupons; other laminates, speckle patterns or camera set-ups are extrapolations; measured interior levels differ by about 2× between specimens in the calibration set; fitted on plain coupons alone c = 1.244, on cutouts alone 1.582.
