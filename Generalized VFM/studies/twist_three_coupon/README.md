# Longitudinal + transverse + edge twist

`python -m studies.twist_three_coupon.make_study` regenerates every artifact in
this folder and writes nothing outside it. Console log in `run_output.txt`.

## Why

`D66` is the two-coupon campaign's blind spot. Neither bending coupon loads the
twist channel, so the stacked answer is **D66 = −3.66 N·mm** — an unphysical
number that only says *no information*. The edge-twist coupon loads `κ_xy`
directly, at 4.5× the next largest channel.

| | D11 | D12 | D22 | D66 |
|---|---|---|---|---|
| two-coupon baseline | 174.35 | 2.07 | 61.43 | **−3.66** |
| **+ edge twist** | 171.46 | 2.34 | 60.58 | **+10.35** |

The bending parameters barely move (≤1.7 %), which is the right behaviour: the
twist coupon adds information where there was none and does not fight the
coupons that were already working.

## The boundary condition

**The loaded edge is clamped into the yoke.** It does not warp and this is not a
free St-Venant twist — the edge rotates as a **rigid body** about the specimen
axis and the machine reports the couple it takes. So

```
EVW = T_x · φ*        (applied couple × virtual rigid rotation)
```

and the BC's job is to make `φ*` well defined. Full derivation in
[`../../../Twist/VFM_export/BC_Twist.py`](../../../Twist/VFM_export/BC_Twist.py).

| edge | treatment | why |
|---|---|---|
| bottom (near the clamp) | `W`, `W_X`, `W_Y`, `W_XY` = 0 | fully sealed: every unknown reaction annihilated. Always admissible; costs only virtual-field freedom |
| top (loaded) | `W_Y` = 0 **and** `W_XY` = 0 | together they make `∂w*/∂y ≡ 0` along the edge, so the unmeasured `M_yy` does no work. `W_Y` alone leaves it wiggling between nodes |
| top | `W_X` **linked** + EXTERNAL | one shared slope = *this edge rotates rigidly*. `φ*` is that linked value |
| top | `W` **free** | required: the rotation must be able to lift one side relative to the other |
| left / right | untouched | the specimen's physical free edges |

`F = −M_xy = −T_x/b`. The distributed group returns `φ*·b`, so `F·φ*·b = T_x·φ*`
— coefficient 1, no factor of two. This is *not* a Kirchhoff distributed
twisting moment on a free plate, and modelling it as one would be the error.

## What the variants show (§5)

| | variant | D66 | verdict |
|---|---|---|---|
| A | shipped | 10.35 | reference |
| B | top `W_XY` free | 10.34 | no effect here |
| C | top `W_X` not linked | 10.42 | the link costs nothing numerically, but it is what makes `φ*` a rotation |
| **D** | **top `W` = 0** | **−3.62** | **collapses to the baseline — the coupon contributes nothing** |
| **E** | **laterals clamped** | **−3.63** | **same collapse — the free edges are load-bearing** |
| F | top `W_Y` free | 10.68 | D66 barely moves, but D12 drops 2.34 → 0.82 |

D and E are the two that matter, and both fail the way the derivation predicts:
clamp `W` on the loaded edge and the rigid rotation cannot happen; clamp the
laterals and the virtual field has nowhere left to live. Neither is a tuning
knob — they are the two places where getting the BC wrong silently deletes the
experiment rather than biasing it.

## Robustness (§3, §4)

* **Weighting**: `gls` 10.35, `block_diagonal` 6.52, `diagonal` 7.22, `ols`
  −5.80. The twist coupon contributes only **two** load steps, so its
  cross-time covariance block is 2×2 — thin, and the ladder shows it.
* **Leave-one-out**: dropping the transverse coupon leaves the system
  unidentifiable. Dropping the longitudinal one keeps D66 (9.23) but destroys
  D11.
* **χ²/dof = 1081, Birge factor 32.9.** Reported σ must be Birge-scaled; eq. (20)
  alone is a variance under a model the data reject.

## The dominant uncertainty (§6)

**The twist coupon's cut angle was never recorded.** `T(90)` leaves the D66
*parameter* untouched, so it is tempting to call the choice free. It is not: the
coupon carries a secondary `κ_yy` at −0.22 of `κ_xy`, and swapping the frame
re-attributes that channel between D11 and D22.

```
twist coupon at  0 deg (shipped)   D66 = 10.35
twist coupon at 90 deg             D66 =  3.81      a factor of 2.7
```

That is larger than the weighting spread, the window sweep and every BC variant
that does not break the identification outright. It is a **missing measurement,
not a modelling choice**, and it should be recovered from the specimen records
before this coupon's D66 is quoted on its own.

## Other limitations, stated

* Two load steps only (steps 3 and 4, referenced to step 2): beyond step 4 the
  twist field stops being self-similar — the `κ_xy` zero crossing migrates 5 mm
  along the specimen across the ramp.
* `SPECIMEN_WIDTH_MM = 9.28` is a placeholder; this coupon's D66 scales as
  1/width.
* `W` free on the loaded cut lets the effective shear do work. The *resultant*
  is bounded — `dFz` is under 1 % of the couple — but a self-equilibrated shear
  distribution is not.

## Outputs

```
data/results.csv       every identification in §2-§4 and §6
data/bc_variants.csv   the table above, with notes
figures/fig1_parameters, fig2_bc_variants, fig3_robustness   (.pdf + .png)
run_output.txt         the full console log
```
