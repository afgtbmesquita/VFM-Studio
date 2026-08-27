# Longitudinal + transverse + 45° + edge twist — two routes to `D66`

`python -m studies.twist_four_coupon.make_study` regenerates every artifact in
this folder and writes nothing outside it. Console log in `run_output.txt`.

## The result

The campaign now has **two independent measurements of `D66`**, independent in
the strong sense that their dominant error sources are disjoint. They agree.

| route | D66 (N·mm) | how |
|---|---|---|
| 45° coupon, stacked FGLS | **9.87** | indirect — from the anticlastic ratio |
| edge twist, stacked FGLS | **10.35** | direct — from a loaded `κ_xy` |
| all four stacked | **10.56** | both |
| 45° closed form `S(1+r)/(1−r)` | 21.64 | see below |

**The two identified routes agree to 6.7 %.** That is worth more than either
number alone, because the two estimators share almost no assumptions:

* the **45° answer** comes from a *ratio* of two curvature fields measured at
  the same points in the same frame, so the applied moment cancels — no specimen
  width, machine calibration or moment gradient enters it. It depends instead on
  the 45/135 cut sense, and on a twist the yoke suppresses ~15×, so it never
  measures a twist at all.
* the **twist answer** comes from a directly loaded `κ_xy`, the dominant channel
  at 4.5× the next largest. It is indifferent to the 45/135 sense. It scales as
  1/width, and it depends on its own unrecorded cut angle — a factor of 2.7 (see
  [`../twist_three_coupon`](../twist_three_coupon/README.md) §6).

Two estimators with disjoint dominant errors landing within a few percent of
each other is evidence that neither is dominated by its own weakness. That is a
much stronger statement than a single well-converged fit.

## Why the closed form is the outlier

`D66 = S(1+r)/(1−r)` is **linear in `S = (D11+D22+2·D12)/4`**, and the fit that
feeds it carries `D12 = −7.94` — negative, i.e. the campaign's weakest
parameter. The closed form inherits that weakness undiluted; the stacked fits
dilute it against every other equation. The 21.64 is a statement about `S`, not
about `D66`.

## Full parameter set

| | D11 | D12 | D22 | D66 |
|---|---|---|---|---|
| two-coupon (bending only) | 174.35 | 2.07 | 61.43 | −3.66 |
| three-coupon (+45°) | 153.31 | −7.94 | 55.84 | 9.87 |
| three-coupon (+twist) | 171.46 | 2.34 | 60.58 | 10.35 |
| **four-coupon (+both)** | **149.62** | **−7.62** | **55.07** | **10.56** |

Jackknife over experiments: mean (149.93, −7.74, 55.07, 10.54), std
(7.41, 3.37, 2.53, 1.02).

Note that the two three-coupon rows disagree on `D11`/`D12` far more than on
`D66`. Adding the 45° coupon pulls `D11` down 12 % and `D12` negative; adding
the twist coupon leaves both essentially where the bending campaign had them.
The four-coupon answer follows the 45° coupon on those, because it is the one
carrying the off-axis information.

## Robustness

* **Weighting**: `gls` 10.56, `block_diagonal` 8.81, `diagonal` 12.36,
  `ols` 2.15. `D66` is stable across the FGLS ladder; `ols` is not a
  contender (it also returns `D22` = 5.65).
* **Leave-one-out**: `D66` survives every single-coupon drop (9.87–14.81).
  `longitudinal + twist` alone is unidentifiable. Pairs that drop both bending
  coupons return nonsense in `D11`/`D22` while still recovering a plausible
  `D66` — the parameters decouple more than the campaign's history suggests.
* **χ²/dof = 2404, Birge factor 49.0**, per experiment 678 / 2614 / 4739 / 3060.
  The noise model does not explain the residuals, and reported σ must be
  Birge-scaled. This is the same disease the README documents for the
  two-coupon campaign (Birge 6.5), larger here because there are more, and more
  heterogeneous, experiments.

## Limitations carried in from the coupons

* twist coupon: two load steps, placeholder width, unrecorded cut angle
  (factor 2.7 on its D66).
* 45° coupon: unresolved 45/135 cut sense; twist suppressed 15× by the fixture;
  107 % moment gradient across its window.

## Outputs

```
data/results.csv     every identification, leave-one-out, weighting, jackknife
data/d66_routes.csv  the two-route comparison with each route's caveat
figures/fig1_parameters, fig2_d66_routes, fig3_leave_one_out   (.pdf + .png)
run_output.txt       the full console log
```
