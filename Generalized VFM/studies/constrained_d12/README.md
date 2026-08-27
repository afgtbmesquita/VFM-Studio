# `studies/constrained_d12/` — the D12 ≥ 0 campaign

Self-contained. **One script generates every study artifact here**, and it writes nothing
outside this folder and modifies nothing outside it:

```
python -m studies.constrained_d12.make_study
```

```
studies/constrained_d12/
├── README.md          this file
├── make_study.py            generates all study artifacts
├── check_loaded_edge.py     proves w* = 0 along every loaded cut
├── constrained_solver.py    compact standalone constrained solve
├── run_output.txt     its full console log
├── data/
│   ├── experiment_setup.csv   BCs, windows, bounds — read from each experiment's own files
│   └── results.csv            every number behind the figures
└── figures/
    ├── fig1_setup                windows + boundary conditions, to scale
    ├── fig2_coefficients         unconstrained vs D12 ≥ 0
    ├── fig3_single_step          one load step per experiment vs all nine
    └── fig4_constraint_cost      what the constraint costs, and whether it is rejected
```

Nothing in the parent folder was changed for this study. The two- and three-coupon campaigns,
their figures and their CSVs are untouched.

---

## 1. The three experiments — boundary conditions and windows

Every number below is read from that experiment's own `.mat` nodes and BC file by
`experiment_setup()`; none is typed in. Same table as `data/experiment_setup.csv`, drawn to
scale in `figures/fig1_setup`.

| | **longitudinal** | **transverse** | **45 deg** |
|---|---|---|---|
| experiment id | 1 | 2 | 3 |
| BC file | `BC_Sample_6.py` | `BC_Transverse_raw.py` | `BC_FortyFive.py` |
| material frame `φ` | **90°** | **0°** | **45°** |
| length runs along | `y` | `y` | `y` |
| **window start → end** | **`y` = −9.9664 → 6.5650 mm** | **`y` = 1.2466 → 18.2071 mm** | **`y` = 5.3020 → 14.5722 mm** |
| length | 16.531 mm | 16.961 mm | **9.270 mm** |
| width | 8.721 mm | 8.373 mm | 8.866 mm |
| **`x` bounds** | [0.892091, 9.613590] | [−0.026368, 8.346780] | [0.054803, 8.920920] |
| **`y` bounds** | [−9.966410, 6.565020] | [1.246570, 18.207100] | [5.302040, 14.572200] |
| area | 144.1789 mm² | 142.0130 mm² | 82.1903 mm² |
| points × steps | 3486 × 9 | 3485 × 9 | 1978 × 9 |
| virtual mesh | 2 × 2 | 2 × 2 | 2 × 2 |
| **FREE edges** | **left + right** | **left + right** | **left + right** |
| **LOADED cuts** | **top + bottom** | **top + bottom** | **top + bottom** |
| clamped DOFs | `top,bottom : W, W_X, W_XY` | `top,bottom : W, W_X, W_XY` | `top,bottom : W, W_X, W_XY` |
| external work | `top,bottom : W_Y` | `top,bottom : W_Y` | `top,bottom : W_Y` |

### Route (B) throughout — every coupon in the frame it was measured in

All three coupons were tested in the **same fixture**, so all three sit the same way: **length
along `+y`, aligned with the machine's `y` axis**, loaded on top and bottom, free on left and
right. Only the **material frame** differs, and only `D` is rotated:

```
φ = 90°   longitudinal   material 1-axis ALONG its length      (the 1-axis is defined by it)
φ =  0°   transverse     material 1-axis ACROSS its length     (cut at 90°, so 1-axis is +x)
φ = 45°   45 deg         material 1-axis at 45° to its length
```

> **Corrected.** An earlier version of this figure used the transverse coupon's *pre-rotated*
> export (`VFM_export/`), whose fields MATLAB turned +90° at export time, and so drew that
> specimen lying on its side with its 1-axis along `+y`. That is not how the test was run.
> This study now uses `TESTS_WITH_45_RAW` — raw fields, `D` rotated — for all three.
> **The identification is unaffected:** the two routes agree to **4.8e-14 relative on θ** and
> **6.4e-15 on the eq. (20) σ**, because internal virtual work is a frame-invariant scalar.
> Doing both would rotate twice; `validate_rotation.py` asserts the equivalence on every run.

**`φ`** is the angle from that experiment's own `+x` axis to the **material 1-axis**,
counter-clockwise. The solver rotates `D` by it (`D_lab = T(φ)ᵀ D_mat T(φ)`); **no field, mesh
or wrench is ever rotated**. The material 1-axis runs along the longitudinal coupon's length.

**The BC form is identical in structure for all three** — the same fixture, described the same
way:

* **free edges** — the specimen's physical lateral edges, left **completely unconstrained**.
  Nothing is attached to them, so `M_nn = 0` and the Kirchhoff shear `V_n = 0` there and both
  boundary work terms vanish on their own. Constraining them would only throw away
  virtual-field freedom. *(On the 45° coupon this is also the mechanism that makes `D66`
  identifiable — the anticlastic ratio is the statement `M_xx = 0`.)*
* **loaded cuts** — `W`, the **tangential** slope and `W_XY` prescribed to zero, so `w* ≡ 0`
  along the edge and the unknown effective shear and twisting moment do no virtual work. The
  **normal** rotation is the only DOF left free, and it is work-conjugate to the measured
  bending moment, supplied as the EXTERNAL group.
* Both cuts are loaded, with `F = [−M, +M]` (`BC_BOTH_CUTS`). On this rig neither individual
  cut moment is known — the machine reports one couple — so the external work is made to
  depend on their **mean**, which the couple does estimate well.

> **Why the 45° window starts at `y` = 5.30 mm and is only 9.27 mm long.** `κ_yy` on that
> specimen reverses sign at `y ≈ 4.2 mm`, so a window reaching further down carries an internal
> moment that changes sign along its length while `F = [−M, +M]` asserts the two cut moments
> are equal. See `../../results/figures/study_45/fig45_trim_sweep` — under this BC form the answer is
> trim-dependent (21 % on `D11`), and sealing the bottom cut instead removes most of that.

---

## 2. The constraint, in closed form

The GLS objective is quadratic, `Q(θ) = (Hθ − b)ᵀ C⁻¹ (Hθ − b)`, so a single inequality
`θ_k ≥ 0` has only two KKT cases — inactive, or active with `θ_k = 0`. With the active set
known the answer is the projection onto `{θ_k = 0}` **in the information metric**:

```
Σ   = (Hᵀ C⁻¹ H)⁻¹                      the eq. (20) covariance
θ_c = θ̂ − (θ̂_k / Σ_kk) · Σ[:, k]
λ   = θ̂_k / Σ_kk                        Lagrange multiplier
ΔQ  = θ̂_k² / Σ_kk                       χ² with 1 dof under H₀: D12 = 0
```

No optimiser, no iteration over active sets. `C` depends on `θ`, so the projection is applied
**inside** the FGLS loop; it settles in 2–3 passes.

### Verification (all asserted in the script, which aborts on failure)

| check | result |
|---|---|
| projection vs deleting column `k` and re-solving | ≤ 3.6e-14 |
| `Σ` from the whitened system vs `result.parameter_covariance` | machine precision |
| `Q(θ_c) ≥ Q(θ*)` **at one fixed C** — the guaranteed inequality | holds in all 11 cases, `ΔQ/Q*` = 0.06 – 0.21 |
| constrained `D12` exactly zero | \|D12\| < 1e-9 |

> **A trap worth recording.** Comparing `χ²/dof` between the free and constrained *FGLS runs*
> is **not** a valid test that the constraint worsens the fit: each run converges to its own
> `C`, so they minimise different functions and either can come out lower — two of the
> single-step cases do exactly that. The inequality that is guaranteed holds at **fixed** `C`,
> which is what `verify_kkt()` checks.

---

## 3. Result — all load steps

| | `D11` | `D12` | `D22` | `D66` | χ²/dof |
|---|---|---|---|---|---|
| **2 coupons**, unconstrained | 174.349 | **+2.074** | 61.428 | −3.663 | 830 |
| **2 coupons**, `D12 ≥ 0` | 174.349 | +2.074 | 61.428 | −3.663 | 830 |
| **3 coupons**, unconstrained | 153.311 | **−7.938** | 55.843 | 9.865 | 2418 |
| **3 coupons**, `D12 ≥ 0` | **143.882** | **0.000** | **54.686** | **9.779** | **2439** |

On two coupons `D12 = +2.07` is already admissible — **the constraint is inactive and changes
nothing**. It only bites once the 45° coupon is in, which is expected: that coupon is the first
test that constrains the combination `D12` lives inside at all.

**The constraint is nearly free.** χ²/dof rises 2418 → 2439, **+0.9 %**. The raw `ΔQ = 4636`
looks like an overwhelming rejection, but it is formed with the eq. (20) covariance the data
reject at χ²/dof = 2418. Birge-scaled — the same correction the parent README applies to every
error bar — `ΔQ/(χ²/dof) = **1.92**`, against `χ²₁,₀.₉₅ = 3.84`: **p = 0.38. The data never
objected to `D12 = 0`.**

**`D66` barely moves: 9.865 → 9.779 (0.9 %).** `D11` absorbs the shift (−6 %), which is exactly
what the 45° coupon's structure predicts — it fixes `S = (D11 + D22 + 2·D12)/4`, so pinning
`D12` forces `D11` down to hold `S` at ≈ 48.

---

## 4. One load step per experiment (3 steps, not 27)

The constraint is **active at every load step** — the unconstrained `D12` runs −12.9 to −25.8.

| step | `D11` | `D12` | `D22` | `D66` | χ²/dof | Birge ΔQ | p |
|---|---|---|---|---|---|---|---|
| 1 | 105.57 | 0 | 69.26 | 8.571 | 1135 | 0.49 | 0.78 |
| 2 | 115.57 | 0 | 51.65 | 9.185 | 3394 | 1.37 | 0.50 |
| 3 | 122.09 | 0 | 54.63 | 7.808 | 5471 | 1.65 | 0.44 |
| 4 | 138.14 | 0 | 55.49 | 8.740 | 6018 | 1.53 | 0.47 |
| 5 | 136.41 | 0 | 58.03 | 7.589 | 6797 | 1.60 | 0.45 |
| 6 | 143.49 | 0 | 51.89 | 8.468 | 9809 | 1.38 | 0.50 |
| 7 | 148.21 | 0 | 58.17 | 8.931 | 9763 | 1.35 | 0.51 |
| 8 | 144.12 | 0 | 51.71 | 8.551 | 10756 | 1.02 | 0.60 |
| 9 | 148.97 | 0 | 55.72 | 8.833 | 9410 | 0.89 | 0.64 |
| **spread** | **34 %** | — | **29 %** | **19 %** | | | |
| **all 9 steps** | **143.88** | **0** | **54.69** | **9.78** | **2439** | 1.92 | 0.38 |

Three things follow.

**`D66` is the most robust coefficient to throwing away 89 % of the data.** 7.59 – 9.19 over
the whole ramp, 19 % spread, **always positive**, mean 8.52 against 9.78 from all 27 steps.
Three load steps recover it to 13 %.

**`D11` drifts systematically with load level**, 105.6 → 149.0, monotone apart from two
reversals. That is not scatter — it is the same load-dependence the campaign already carries
in `D11`, now visible one step at a time. Only the highest steps approach the all-steps value,
which is what a 27-step fit weighted toward the larger signals should give.

**`D12 = 0` is never rejected at any load level.** Birge-scaled `ΔQ` = 0.49 – 1.65, all below
3.84 (`figures/fig4_constraint_cost`, right panel). The single load step that comes closest to
objecting is step 3, at p = 0.44.

χ²/dof grows ~10× along the ramp (1135 → 10756) in both the free and constrained fits. The
model misfit grows with load faster than the noise model does — consistent with the parent
README's finding that σ tracks the applied load, and a reminder that these χ² are *under the
noise model*, not goodness-of-fit in an absolute sense.

---

## 5. What to quote

```
D11 = 143.9      D12 = 0 (constrained)      D22 = 54.7      D66 = +9.8   N.mm
```

with the caveats that already apply and are unchanged by this study: `D11` carries a 21 %
cut-sense ambiguity and a 146 % BC-form spread on the longitudinal coupon; `D66 > 0` rests on
the free-edge treatment (`../../README.md`, the BC-study section). `D12 ≡ 0` is the defensible
reporting choice — physically admissible, costs 0.9 % of fit, not rejected at p = 0.38, and it
leaves `D66` and `D22` essentially untouched.
