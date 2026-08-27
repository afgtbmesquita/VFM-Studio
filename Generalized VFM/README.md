# Generalized VFM — identification by Feasible Generalized Least Squares

A copy of the plate VFM solver with the identification loop replaced. The noise covariance
is now **measured from the fields**, **propagated through the virtual fields**, and carried
**inside the convergence loop** alongside the parameters.

```
python -m workflows.validate_gls                  # known-answer benchmark -- READ THIS FIRST
python -m workflows.validate_rotation             # rotating D == rotating the fields
python -m workflows.run_campaign_gls --all        # the real identification (2 coupons)
python -m workflows.run_campaign_45 --all         # + the 45 deg coupon
python -m workflows.bc_study_45                   # its BCs and window trim -- READ THIS
python -m workflows.make_figures --with-45 --bc-form both-cuts
python -m workflows.make_figures                  # the paper figures
python -m workflows.make_raw_frame_transverse     # raw-frame transverse input
python -m workflows.make_gui_project              # build + verify the GUI project
python run_vfm.py                                  # GUI: load projects/campaign.vfmproj.json
```

## Repository map

| path | contents |
|---|---|
| `vfm_*.py` | reusable solver, constitutive, mesh, covariance, identification, and project modules |
| `run_vfm.py` | interactive/headless application entry point |
| `bc_gui/` | boundary-condition editor and results GUI |
| `workflows/` | campaign runs, validation, sensitivity studies, audits, project generation, and figure generation |
| `projects/` | loadable `.vfmproj.json` run definitions |
| `boundary_conditions/` | boundary-condition files created by the GUI |
| `tests/` | solver smoke tests; run with `python -m tests.smoke_test` |
| `examples/` | small inputs used for documentation and verification |
| `studies/constrained_d12/` | self-contained `D12 ≥ 0` study and its own data/figures |
| `results/two_coupon/` | two-coupon tables and captured console output |
| `results/three_coupon/` | three-coupon tables and captured console output |
| `results/audit/` | assumption-audit data and captured console output |
| `results/bc_studies/` | machine-readable boundary-condition study tables/logs |
| `results/figures/` | all generated publication figures |

Run commands from this directory. Generated files stay under `results/`, saved run
definitions stay under `projects/`, and source modules no longer share folders with outputs.

Same GUI, same ABD parameter setup, same BC files. The original solver at
`G:\…\Inverse_Solution\plate` is untouched.

---

## Boundary conditions used

Both coupons, in their own measurement frame, run along **y** and are loaded on the **top** and
**bottom** cuts. Lateral edges (`left`/`right`) are the specimen's traction-free edges and
appear nowhere.

| | top cut | bottom cut |
|---|---|---|
| `W` (deflection) | **= 0** | **= 0** |
| `W_X` (tangential slope) | **= 0** | **= 0** |
| `W_XY` (twist) | **= 0** | **= 0** |
| `W_Y` (normal rotation) | **free** → does external work | **= 0** *(top-only)* / free *(both-cuts)* |
| EXTERNAL groups | 1, `W_Y`, distributed, sort `x` | none *(top-only)* / 1 *(both-cuts)* |

- `W`, `W_X`, `W_XY = 0` on both cuts annihilates the **shear** and the **twisting moment**.
- **top-only** *(now the default)* adds `W_Y = 0` on the bottom cut, annihilating its
  **unknown reaction moment** too. Only the top region does external work.
- **both-cuts** *(previous)* leaves `W_Y` free at both ends, so the moment works at both, with
  `F = [-M, +M]`.

Select with `build_campaign(..., bc_form=BC_TOP_ONLY | BC_BOTH_CUTS)` or
`make_figures.py --bc-form`.

## Material frame

**The material 1-axis runs along the LONGITUDINAL coupon's length**, so that coupon identifies
`D11` and the transverse one `D22`.

| coupon | `material_orientation_deg` | identifies |
|---|---|---|
| longitudinal | **90** (length is the lab +y axis) | `D11` |
| transverse, raw frame | **0** (cut at 90°, so 1-axis is across it) | `D22` |
| transverse, pre-rotated export | 90 (shares the longitudinal frame) | `D22` |

Changing the convention is an exact `D11`↔`D22` relabelling — verified to **5e-14** against the
previous run. `T(180) = I`, which is why shifting the whole frame by +90° leaves the raw
transverse coupon at 0 rather than 180.

## The result

Figures and `figure_data.csv` are built on the **top-only** form; `results/figures/both_cuts/` holds
the same five figures for the other form.

| | `D11` (longitudinal) | `D12` | `D22` (transverse) | `D66` | χ²/dof |
|---|---|---|---|---|---|
| **top-only** *(default)* | **120.17** ±6.96 | 0.44 ±2.05 | **63.32** ±3.13 | −4.10 ±2.05 | **285** |
| both-cuts | 174.35 ±12.37 | 2.07 ±4.33 | 61.43 ±2.47 | −3.66 ±3.54 | 830 |

Birge-scaled intervals (eq. (20) σ × √(χ²/dof)); both converge in 5 iterations.

**`D22` barely moves (61.4 → 63.3, inside its own interval); `D11` drops 31 %.** That is the
expected signature: the transverse coupon is near pure bending (9 % moment gradient) so its
coefficient is BC-insensitive, while the longitudinal coupon is not (111 %) and its coefficient
is not. **`D11` is the coefficient this campaign cannot pin down.**

## Boundary-condition study

`python -m workflows.bc_study` → `results/figures/figF_bc_study.pdf` +
`results/bc_studies/two_coupon/bc_study.csv`.

Rule: a cut is *sealed* when `W`, `W_X`, `W_XY`, `W_Y` are all prescribed. A DOF may be left
free only if its conjugate moment is **measured** and supplied as an EXTERNAL group. Top cut =
measured couple; bottom = unmeasured reaction; left/right = free edges, never touched.

| | variant | `D11` | `D12` | `D22` | `D66` | χ²/dof |
|---|---|---|---|---|---|---|
| A | reference (all rules) | 120.17 | 0.44 | 63.32 | −4.10 | 285 |
| B | both cuts loaded | 174.35 | 2.07 | 61.43 | −3.66 | 830 |
| C | bottom `W_XY` free *(leaks)* | 117.17 | −0.95 | 62.09 | −3.46 | 298 |
| D | bottom `W_X` free *(leaks)* | 119.40 | 0.76 | 63.08 | −3.97 | 299 |
| E | top `W_XY` free *(leaks)* | 115.34 | −5.86 | 62.69 | −4.98 | 393 |
| F | top `W_X` free + measured `M_xy` | 119.88 | 1.29 | 60.30 | −7.09 | 444 |
| G | **link `W_Y` across top** | 122.16 | 1.07 | 63.66 | −2.99 | **247** |
| H | link `W_Y` top + bottom `W_XY` free *(leaks)* | 119.02 | −0.14 | 62.50 | −2.63 | 256 |

Findings:

- Across the four **admissible** variants (A, B, F, G) the spread is **45 % on `D11`** but only
  **5 % on `D22`**. The BC choice is consequential exactly where the pure-bending assumption
  already fails.
- **G (linking) gives the best fit of all**, χ²/dof = 247, and lands within 2 % of A. Tying
  `W_Y` across the top shrinks the virtual-field space without leaking any traction — a cheap,
  legitimate improvement.
- The leaks (C, D, E, H) shift `D11` by 1–4 % but wreck `D12`: E drives it to −5.86.
- F supplies `M_xy` from the machine `Tx`. Its sign convention is not fixed for this frame;
  flipping it moves `D66` from −7.1 to −2.7 but `D11` by 1.6 % and `D22` by 0.6 %, so it matters
  only for the coefficient that is unidentifiable anyway. `|M_xy|` is 0.23–0.38 N against
  `|M_yy|` of 0.25–2.5 N — **not** negligible at the first load steps.

The spread among admissible variants is model error: each is a legitimate virtual field, so no
variance formula can see the difference.

**Quote these two only.** For an interval, see below — and read the distinction first,
because it is the main caveat on the whole result.

### σ under the model ≠ σ of the data

Eq. (20) gives ±0.086 and ±0.429. Those numbers answer one question:

> *if the only error were spatially white DIC noise with covariance `Γ̂`, how far would `θ̂`
> move?*

They do **not** answer "how well is `D_ij` known?". The data reject that noise model at
**χ²/dof = 830** — it accounts for **0.12 %** of the observed residual variance. So eq. (20)
is a variance *conditional on a model that is demonstrably wrong*, and it is not plotted
against the coefficients anywhere (see `fig1`, which deliberately carries no error band, and
`fig3`, which is titled for this distinction).

A defensible interval is at least √(χ²/dof) = **29×** those values:

```
D11 = 61.4 +/- 2.5 N.mm     D22 = 174.3 +/- 12.4 N.mm
D12 =  2.1 +/- 4.3 N.mm     D66 =  -3.7 +/-  3.5 N.mm     <- both consistent with zero
```

and even that assumes the misfit is a pure *scale* error rather than a *shape* error, which
is not established. Treat it as a floor, not an uncertainty budget.

`D66 = −3.66` is **not a measurement**. It is negative, which no real laminate can be (the D
matrix must be positive definite), and the two coupons disagree about it by ~7 N·mm — larger
than |D66| itself.

### No κ, and why

An earlier version multiplied `C` by κ, the measured number of effectively correlated DIC
points, to patch the spatial-whiteness violation. **That has been removed from every
reported result.** κ is an approximation to eq. (1) that assumes the virtual fields are
constant across the correlation neighbourhood, and it needs a plateau chosen off a
block-size ladder — two judgement calls. A covariance the derivation does not contain has no
business setting a published error bar, and absorbing a misfit into the weighting hides the
thing worth reporting.

The honest repair is to evaluate **eq. (1)** itself. Until then the solver stays loyal to
eq. (3) (κ = 1) and the misfit is *reported* via χ²/dof rather than absorbed.

Dropping κ costs nothing and gains accuracy: the Birge-scaled interval is essentially
unchanged (D11 ±2.47 against ±2.57 with κ; D66 ±3.54 against ±3.52), and the agreement with
the independent single-test values actually **improves** — `D11` 3.1 % (was 3.5 %) and
`D22` 0.8 % (was 2.8 %). κ was not doing work the goodness-of-fit statistic does not do
more honestly.

The spatial correlation is still measured and still reported — `figD_spatial_whiteness`
shows ρ = 0.92 at one grid step against the δ_pq that eq. (2) assumes. It is a diagnostic of
how wrong the model is, not a correction to it.

Each has an independent single-test cross-check, and FGLS agrees with both more closely
than the previous inverse-variance pooling did:

| | FGLS | single test | gap | inverse-variance gap |
|---|---|---|---|---|
| `D11` | **61.43** | 63.41 (transverse alone) | **3.1 %** | 4.9 % |
| `D22` | **174.35** | 173.03 (longitudinal alone) | **0.8 %** | 3.3 % |

Full side by side:

| run | `D11` | `D12` | `D22` | `D66` |
|---|---|---|---|---|
| longitudinal alone (OLS) | 4.13 | 0.52 | 173.03 | −9.92 |
| transverse alone (OLS) | 63.41 | 0.12 | −0.97 | −2.66 |
| stacked naive (OLS) | 32.27 | −0.36 | 85.46 | −7.86 |
| stacked inverse-variance | 60.33 | −0.16 | 178.73 | −6.47 |
| FGLS block-diagonal | 61.42 | 2.39 | 175.72 | −5.73 |
| **FGLS (this method, eq. 3 verbatim)** | **61.43** | 2.07 | **174.35** | −3.66 |
| *[sensitivity only, not reported]* FGLS with κ | *61.19* | *1.52* | *177.96* | *−4.04* |

`D12` and `D66` are **still not recovered**, and this is not a shortcoming of the estimator.
Both coupons bend into a near-cylindrical shape, so the secondary curvature is at noise
level and `D12` — which multiplies exactly that curvature — has almost nothing to act on.
`workflows/validate_gls.py` reproduces the same failure on a synthetic built at these coupons' *own
measured biaxiality*. Recovering them needs an off-axis (±45°) coupon, a torsion test, or
genuinely biaxial bending.

The solver is now ready for that off-axis coupon — see *Frames: rotate `D`, not the fields*
below. In the material frame an orthotropic `D` decouples the twist exactly
(`Mxy = D66·κxy`), and a 45° pure-bending test applies `Mxy = −M/2`, so `D66` gains a
sensitivity direction that neither existing coupon touches. The data does not exist yet; the
code path does, and it is the same one the 90° coupon now runs through.

---

## Frames: rotate `D`, not the fields

The transverse coupon was cut at 90°, and the campaign used to reconcile it with the
longitudinal one by **rotating its measured fields** — MATLAB rotates the node coordinates
and the curvature tensor +90° at export
(`SCRIPT_export_transverse_ML_dataset.m:319-337`). That works for 90°, where the rotation is
a signed permutation of channels. It does **not** extend to 45°, where a rotation *mixes*
`κxx`, `κyy` and `κxy` instead of permuting them — so the `cxy` factor-of-two and sign, today
harmless because that channel carries no information, would contaminate all three.

Internal virtual work is a frame-invariant scalar,

```
κ*_labᵀ D_lab κ_lab  =  κ*_matᵀ D_mat κ_mat        with  κ_mat = T(φ) κ_lab
⇒  D_lab = T(φ)ᵀ D_mat T(φ)
```

so there are exactly **two** self-consistent formulations, and everything must live in one
frame:

| | fields | mesh | wrench / BCs | `D` |
|---|---|---|---|---|
| (A) the old route | rotated to material frame | rotated | rotated | orthotropic, unrotated |
| (B) **now the default** | **raw test frame** | **raw** | **raw** | `T(φ)ᵀ D_mat T(φ)` |

**Doing both would rotate twice.** Under (B) the measured fields are used exactly as
exported — that is the whole point.

### The factor of two

`κxy` is the **engineering** twist (`vfm_mesh.py:452`, `B[:,2,:] = -2·N_xy`), so the naive
tensor rotation `T_σ D T_σᵀ` is wrong in the 16/26/66 entries. The correct map is the
CLT/Reuter form `T = R T_σ R⁻¹` with `R = diag(1,1,2)`:

```
T(φ) = [[  c²,   s²,   cs   ]
        [  s²,   c²,  -cs   ]
        [-2cs,  2cs, c²-s² ]]
```

`rotate_constitutive_model()` implements this. Two known cases pin the convention, both
asserted: φ=90° must give the D11↔D22 swap, φ=45° must match Jones's closed forms
(`D_lab[0,0] = (D11+D22+2D12)/4 + D66`, `D_lab[2,2] = (D11+D22−2D12)/4`,
`D_lab[0,2] = (D11−D22)/4`).

### Why the solver needed no changes

`D_lab` is **linear** in the entries of `D_mat`, so the rotated basis matrices are still
constant — `basis_lab_k = Tᵀ basis_mat_k T` — and `D(θ) = Σ_k θ_k basis_k` survives untouched.
`θ` stays a single **material-frame** vector shared by every experiment, which is exactly what
the GLS stacking needs; only `material_matrix` and `basis_matrices` move. No numeric consumer
indexes individual entries of either, so a dense `D` (with D16/D26 present) flows through
`vfm_optimal_field`, `vfm_noise_covariance` and `vfm_mesh` unchanged.

Note that this is a **congruence**, not a similarity: eigenvalues *do* move (6 → 12,
174.4 → 139.5 at 45°). Positive-definiteness survives by Sylvester's law of inertia, which is
what the solver actually needs, but don't read the conditioning as invariant.

### The proof

`python -m workflows.validate_rotation` identifies the transverse coupon **both ways** and requires the
two to agree — `workflows/make_raw_frame_transverse.py` builds route (B)'s input by undoing the export's
rotation exactly (a signed permutation inverts losslessly; the script refuses to write unless
its own round trip is bit-exact).

| check | result |
|---|---|
| φ=90° gives the D11↔D22 swap | 9.8e-15 |
| φ=45° matches Jones's closed forms | 4.3e-14 |
| `D(θ) = Σ θ_k basis_k` after rotation | 2.8e-14 |
| internal virtual work frame-invariant | 2.3e-13 |
| constraint nullity, route A vs B | 18 = 18 |
| **transverse alone, FGLS: A vs B** | **1.2e-15** |
| **transverse alone, OLS: A vs B** | **3.4e-16** |
| **full campaign, FGLS: A vs B** | **2.8e-15** |

The agreement is at machine precision, and the constraint nullity matching confirms both
routes build the *same* discrete virtual-field space rather than two that happen to agree.

`projects/campaign.vfmproj.json` now ships on route (B): raw transverse fields with
`material_orientation_deg = 90`. `workflows/make_gui_project.py` still compares it against
`campaign_common`'s route-(A) run and refuses to leave the file on disk unless they match — so
the equivalence is re-proved every time the project is rebuilt (currently 2.5e-13 on θ and
6.8e-14 on the eq. (20) σ). **`D11 = 61.43` and `D22 = 174.35` are unchanged.**

### Using it

`material_orientation_deg` is the angle from an experiment's own measurement frame (+x) to the
material 1-axis, counter-clockwise positive. It is a **per-experiment** property — one coupon
in one fixture has one orientation for the whole load ramp — carried on `ProjectTest` and
`IdentificationCase`, and `check_experiment_consistency()` rejects a run where load steps of
one experiment disagree. It defaults to `0.0`, and a zero angle returns the constitutive model
**unchanged**, so every pre-existing project and every file written before this field existed
reproduces bit-identically (no format version bump — `load_project` reads it with a default).

In the GUI it appears in two places, kept in sync: the experiment tab's "Shared by every time
step" box, and the BC editor's Solver Settings tab. The BC editor draws the material **1** and
**2** axes on the mesh, live, so the frame `θ` is expressed in is visible while boundary
conditions are being assigned rather than being a number to hold in your head.

**A 45° coupon needs nothing further** — it is this code path with a different angle.

---

## Why the moment is applied at both ends

The BC files put an EXTERNAL group on **both** loaded cuts, with `F = [-M, +M]`. Neither cut is
fully clamped: `W`, the tangential slope and the twist are prescribed to zero there, but the
**normal rotation is free**, so the applied moment does virtual work at both ends.

That is not double counting. The two edge rotations are opposite, so the two work terms add to
`M·Δθ*` — the total relative rotation, which is what a pure-bending moment is conjugate to.

The stricter alternative — also prescribe `W_Y = 0` on the cut whose moment is **unknown**, so
that cut's reaction does no virtual work, leaving one EXTERNAL group carrying the measured
moment — is the textbook treatment whenever one end is an uninstrumented reaction. It is built
by `clamp_variant()` in `workflows/compare_bc_forms.py`:

```
python -m workflows.compare_bc_forms                     # the table below
python -m workflows.compare_bc_forms --write-projects    # loadable clamped-form projects
```

### On the real coupons

| longitudinal (identifies `D22`) | D11 | D12 | **D22** | D66 |
|---|---|---|---|---|
| **OLS** — curvature varies **111 %** end to end | | | | |
| (a) moment at both cuts *[shipped]* | 4.130 | 0.515 | **173.03** | −9.925 |
| (b) clamp bottom, load top | 3.408 | 0.520 | **121.30** | −7.401 |
| (b) clamp top, load bottom | 2.892 | 0.277 | **374.77** | −3.685 |
| **FGLS** | | | | |
| (a) moment at both cuts | 1.259 | −0.016 | **181.70** | −9.157 |
| (b) clamp bottom, load top | 0.774 | −0.032 | **131.09** | −6.367 |
| (b) clamp top, load bottom | 0.529 | −0.270 | **347.79** | −1.765 |

| transverse (identifies `D11`) | **D11** | D12 | D22 | D66 |
|---|---|---|---|---|
| **OLS** — curvature varies **9 %** end to end | | | | |
| (a) moment at both cuts *[shipped]* | **63.41** | 0.118 | −0.967 | −2.660 |
| (b) clamp bottom, load top | **65.34** | 0.137 | −1.546 | −3.495 |
| (b) clamp top, load bottom | **61.13** | 0.033 | −0.313 | −1.163 |
| **FGLS** | | | | |
| (a) moment at both cuts | **61.19** | −0.012 | 0.194 | −2.008 |
| (b) clamp bottom, load top | **63.73** | −0.014 | 0.096 | −1.816 |
| (b) clamp top, load bottom | **57.95** | −0.009 | 0.072 | −0.699 |

**The one-cut spread tracks the curvature gradient, coupon for coupon:**

| coupon | curvature varies end to end | one-cut spread |
|---|---|---|
| longitudinal | 111 % | **253.5 N·mm — 146 % of the value** |
| transverse | 9 % | **4.2 N·mm — 7 % of the value** |

On both coupons the two one-cut answers **bracket** the both-cut answer.

### Why, and which to use

Form (b) is the correct formulation *when one cut's moment is known and the other is not*. On
this rig **neither individual cut moment is known**: the machine reports a single couple
(`dTy`), and the export writes it to both cuts as `F = [-M, +M]`, which *assumes* equilibrium.
The longitudinal README's Finding 3 measures the violation — an unmeasured dead weight (~170 g
of grip and fixture outboard of the load cell) makes the internal moment differ between the two
cuts by ~100 % of the applied couple, while every force channel reads ≤ 0.1 N.

With a virtual field antisymmetric along the length, form (a)'s external work depends on the
**mean** of the two cut moments — which the machine couple does estimate well. Each form (b)
depends on a **single** cut moment, which it does not. That is exactly why the longitudinal
one-cut answers land at 121 and 375 around 173, reproducing the figures the longitudinal README
already records.

So:

- **Longitudinal — use (a).** Form (b)'s premise is violated by ~100 % here; its 146 % spread is
  a measurement of the moment error, not an uncertainty to average away.
- **Transverse — either.** The gradient is 9 %, the spread 7 %, and (a) sits between. This
  coupon is close enough to pure bending for the stricter form to be usable.
- **A future rig with one end instrumented and the other a reaction — use (b).** That is what
  it is for, and `--write-projects` emits ready-to-run projects.

**Take this seriously when quoting `D22`.** The spread between defensible virtual fields on the
longitudinal coupon is 146 % — vastly larger than the Birge-scaled ±12.4 N·mm (7 %) in the
uncertainty ladder below, which like every entry there is a *variance* and cannot see it. The
fix Finding 3 names — weigh the fixture outboard of the load cell, or move the DIC window to
where the curvature is flat — is what would make `D22` quotable at better than tens of percent.

---

## Two guards the GUI route was missing

`workflows/run_campaign_gls.py` has always called `assert_orientation()`; the project/GUI route
**bypassed it entirely**. Two failures were therefore silent there.

**A run with no EXTERNAL group.** `build_external_work_matrix()` returns a single
*placeholder* column when the EXTERNAL list is empty, so the solver reported

```
Applied force from the measurement file has 2 entries but the
BC file declares 1 external-work group(s).
```

The count was **zero**, not one — the message sent you hunting for a size mismatch that did
not exist. `check_experiment_consistency()` also let it through, because it only complained
when *both* `external` and `prescribed` were empty. Now: the consistency check names it before
the run, the solver message says the count is zero rather than one, and the experiment dialog
asks before discarding EXTERNAL groups the BC editor came back without.

**A load declared on the wrong axis.** An EXTERNAL group on `W_X` when the coupon bends in
`κyy` produces a run that completes happily and reports the **other** stiffness coefficient —
which looks exactly like "the results came out swapped". `check_load_orientation()` now
compares the declared load DOF against the measured dominant curvature and warns, in the
headless run and as a confirmation dialog in the GUI.

It is **advisory, never blocking**, and stays silent below a 5× dominance ratio — a genuinely
biaxial coupon (an off-axis or torsion test) has no dominant curvature, and the check must not
cry wolf on the 45° experiment this solver now supports.

### Verified

| | |
|---|---|
| shipped campaign, both checks | clean — no false positives |
| EXTERNAL groups wiped | caught before the run; solver message now says "zero, not one" |
| load DOF moved to the wrong axis | caught, naming the measured dominant curvature |
| biaxial (45°-like) coupon | **silent** |
| **GUI dialog route vs headless** | **0.000e+00 on θ and on the eq. (20) σ** |

---

## Applied load: per experiment, per time step

Each experiment now declares **where its applied load comes from**, and the choice is
per-experiment — one coupon in one fixture takes its loads from one place.

| `applied_load_source` | behaviour |
|---|---|
| `"mat"` (default) | each step's load is read from the `F` vector in its own `.mat`; the declared `value` entries are the fallback |
| `"manual"` | the per-time-step `EXTERNAL` `value` entries win; `F` is the fallback |

**Why this was needed.** The load columns in the experiment dialog were being read, stored in
the project, written to JSON — and then **silently ignored** at solve time, because
`_resolve_applied_force` always preferred the `.mat` `F`. Every dataset in this campaign ships
an `F`, so typing a load did nothing and said nothing. A new coupon whose `.mat` carries `F`
but whose true moment you want to supply by hand was simply not expressible.

**A second bug, deeper.** `_PreparedExperiment` resolved *every* load step against
`self.boundary_conditions` — the **first** step's. So per-step values collapsed to step 0's
across the whole ramp. Each step now resolves against its own `boundary_conditions`; the
external-work *operator* is still shared, because that is topology and is exactly what makes
those steps one experiment. `_check_homogeneous()` now also rejects steps that disagree on
EXTERNAL group count or load source.

**All-or-nothing, on purpose.** A partially filled table falls back to `F` *wholesale*, never
group by group — blending a typed load on one edge with the file's on another would silently
describe a different experiment from the one you meant. `check_experiment_consistency()`
catches blank cells before the run and names the step and the group.

In the GUI, each experiment tab has an **Applied load from** selector. On `.mat` the load
columns are greyed and read-only, with a note saying they are not in use; on manual they
become editable, blank cells are highlighted, and the note counts how many are still empty and
states that the run will be refused until they are filled.

Verified: retyping a step's own `F` by hand reproduces the baseline **bit-identically**;
doubling every typed load scales θ by **exactly 2.0** across all four coefficients (`Kθ = U`
with `U ∝ F`); and scaling a **single** step changes the answer, which under the old code it
could not have.

Default is `"mat"`, so every existing project, the shipped campaign, and files written before
this field existed all reproduce unchanged.

---

## What changed, and why

`../stacked VFM solution` established that ordinary least squares is the wrong estimator
here: with Pierron's speciality condition every `H_i` is the identity, so the stacked
least-squares collapses to the plain arithmetic mean of the per-load-step estimates, and a
test carrying no information about a parameter still votes with full weight. It fixed that
with per-parameter inverse-variance weights `1/(σ_i η_ik)²`. Three things were left open:

| | stacked VFM | this folder |
|---|---|---|
| weights | scalar per parameter | full `m×m` matrix per load-step pair |
| `σ_i` | **assumed equal** between tests | **measured** from the fields |
| load steps | treated as independent estimates | cross-time covariance measured and used |
| `ΔA` | omitted | carried (eq. 4) |

### The load steps are not independent — this is the main finding

Measured cross-step correlation of the residual field, mean over the off-diagonal:

```
longitudinal   0.769          transverse   0.636
```

Nine load steps therefore carry closer to **one or two steps' worth** of independent
information, not nine. The previous pooling assumed 0.0 here. That does not move the
central estimate much, but it is why its error bars were far too tight.

### The two tests have genuinely different noise

The old `σ_T/σ_L = 1` assumption is now unnecessary — it is measured per load step and per
kinematic component. Measured residual σ (×10⁻³, each test's **primary** curvature —
κyy longitudinal, κxx transverse — first→last load step):

```
longitudinal   1.12 -> 1.75          transverse   0.37 -> 1.01
```

So the transverse fields really are the quieter ones — by ~3× at the first load step and
~1.7× by the last, since the transverse noise grows faster with load. The old default knew
neither the ratio nor its drift across the ramp.

---

## The method

Notation follows the derivation. `n_g = 3` here (κxx, κyy, κxy) because the campaign is
`out_of_plane`; the code is generic in `n_g ∈ {3, 6}`.

**1. Estimate Γ_i, once, outside the θ loop.** Fit a smooth spatial basis to the measured
kinematics at each load step and treat what it cannot represent as fluctuation:

```
R_{i,t} = (I - Q Qᵀ) Y_{i,t}          Z_i = [R_{i,1} … R_{i,T}]
Γ_i     = Z_iᵀ Z_i / ν_i              (27 × 27 here; PSD by construction)
```

**2. Propagate through the actual virtual fields.**

```
C_{i,ts} = U_tᵀ [ Σ_p w_p² B_pᵀ K Γ_{i,ts} Kᵀ B_p ] U_s          (eq. 3, verbatim)
```

Nothing assumes `E*_t = E*_s`: every block is built from the virtual fields actually used at
steps `t` and `s`. Implemented via a θ-independent tensor `Φ_ef = Σ_p w_p² B_p[e]ᵀ B_p[f]`
built once per experiment, so the per-iteration cost is a small `n_g × n_g` contraction.

**3. Solve, without ever forming C⁻¹.**

```
C_G = blockdiag(C_1, C_2)     dense within an experiment, independent between
L   = chol(C_G)               θ = lstsq(L⁻¹H_G, L⁻¹b_G)
```

Converged on **both** `‖Δθ‖/‖θ‖` (eq. 18) and `‖ΔC‖_F/‖C‖_F` (eq. 19). Typically 5
iterations.

### The `ΔA` factor

`_assemble_virtual_stiffness_matrix` uses the integration weight `w_p` to the **first**
power, so `η² = uᵀGu` is short one factor of `ΔA` relative to a true variance. The
covariance here carries `w_p²`, i.e. eq. (4)'s `ΔA`. Since `ΔA` differs between the two
experiments (`w_long/w_trans = 1.0149`) this is a real, if small, reweighting — and the GLS
form is the correct one. `workflows/validate_gls.py` pins it down both ways.

---

## Spatial correlation — measured, reported, never applied

The derivation assumes `E[η_p η_qᵀ] = δ_pq Γ`. **On this data that is strongly violated**:
the residual correlation at one grid step (0.20 mm) is **0.92 (longitudinal) and 0.94
(transverse)**, decaying over ~1 mm, because Vic-3D curvatures are differentiated and
smoothed fields.

This is reported (`figD_spatial_whiteness`) and **not corrected**. See *No κ, and why* above.
The consequence is carried instead by the goodness-of-fit statistic: with eq. (3) taken
verbatim, χ²/dof = 830.

For the record, the size of what eq. (3) omits: κ = 10.5 (longitudinal) and 18.5
(transverse), estimated by spatial block averaging. Applying it would inflate σ by ~3.8×
and reduce χ²/dof from 830 to 42 — so it accounts for roughly a factor of 20 of the misfit,
and a factor of 42 would survive it. `GlsSettings(spatial_correction=True)` reproduces that
variant as a labelled sensitivity check; it is never used for a reported number.

Note also that because every `H_i` is the identity, a covariance factor **common to both
experiments cancels exactly from `θ̂`**. Only the ratio between experiments could move the
answer, which is why dropping κ shifts `D22` by only 2 %.

## Uncertainty: what each number actually answers

All values N·mm, from `workflows/audit_assumptions.py` (150-replicate bootstraps).

| source | `D11` | `D12` | `D22` | `D66` | answers |
|---|---|---|---|---|---|
| eq. (20), eq. (3) verbatim | 0.086 | 0.150 | 0.429 | 0.123 | *how far would θ̂ move under spatially white DIC noise?* |
| bootstrap, white noise | 0.297 | 0.344 | 0.660 | 0.139 | the same question, without the first-order approximation |
| bootstrap, measured spatial correlation | 0.698 | 0.916 | 1.640 | 0.464 | as above, with eq. (2) relaxed |
| basis degree 2–10 spread | 0.214 | 0.622 | 1.546 | 0.230 | how much the Γ estimator's own tuning matters |
| jackknife over load steps | 2.196 | 1.293 | 3.528 | 0.344 | how much one load step is worth |
| **eq. (20) × √(χ²/dof) = ×28.8** | **2.474** | **4.326** | **12.372** | **3.541** | **a floor on the real interval** |

**Eq. (20) is not wrong.** The white-noise bootstrap reproduces it to a factor of 1.1–3.5
under eq. (2)'s own noise model, so the algebra and the first-order expansion are sound. It
is simply conditional on a noise model the data reject at χ²/dof = 830.

The last row is the Birge correction: rescale the reported uncertainty until the fit is
consistent. Unlike κ it introduces nothing into the *model* or the *estimator* — θ̂ is
untouched — it only stops the uncertainty being reported as smaller than the observed
misfit. It remains a **floor**, because it assumes the misfit is a pure scale error; if the
model error has structure (which it does — see figC panel a, where `D11` misfits 240× worse
than `D12`), a single scalar cannot describe it.

---

## Verification

`python -m workflows.validate_gls` — 13 checks, all passing:

| check | result |
|---|---|
| covariance kernel vs the solver's own `G` | 5.8e-15 relative |
| `weighting='ols'` reproduces `run_identification_cases` | 7.9e-14 |
| `weighting='diagonal'`, `Γ = δ_ts I` reproduces inverse-variance pooling | 1.3e-07 |
| `Γ` recovered from injected noise | 4.1 % |
| temporal correlation recovered | 0.805 against 0.800 imposed |
| per-component σ recovered | 0.98–1.00 |
| `κ ≈ 1` on genuinely white noise | 1.000 |
| synthetic biaxiality matches the real coupons | 1.18× / 1.36× |
| FGLS recovers `D11`, `D22` (noise-free) | 0.40 %, 0.00 % |
| **FGLS recovers `D11`, `D22` (correlated noise)** | **0.10 %, 0.02 %** |

On that same synthetic the naive stacked solve is **36 %** and **54 %** wrong. The benchmark
is deliberately built at the coupons' own measured biaxiality so it is no easier than the
real experiment — `assert_representative()` enforces that, so a change to `D_TRUE` cannot
silently make the test easy.

Also passing: `python -m tests.smoke_test` (24 checkpoints) and `python -m bc_gui._smoke_test`.

> `tests/smoke_test.py` shipped **broken** in the original solver: a loop at what is now line 338
> rebound `model`, clobbering the iso/both model that section 5 needs, and the run died with
> `Mesh analysis_type 'both' does not match constitutive analysis_type 'out_of_plane'`. The
> original fails identically — this is not a regression from the copy. Fixed here by renaming
> the loop variable; no solver behaviour changed.

### The twelve validation items

| # | item | verdict |
|---|---|---|
| 1 | spatial-white assumption | **VIOLATED**, ρ ≈ 0.92–0.94 at one grid step. Corrected; both paths reported |
| 2 | spatial homogeneity of the noise | assumed, not tested per-region. Anisotropy measured (0.73 vs 0.89) and handled by the non-parametric κ |
| 3 | signal/noise separation | **PARTIAL.** σ correlates with load at 0.88–0.99 but σ/F falls ~4× — a noise floor plus a modest lack-of-fit component. `Γ̂` is *effective error*, not pure DIC noise |
| 4 | sensitivity to the smoothing basis | **PASSES.** Degree 2→10 moves `D11` 0.95 %, `D22` 2.0 % — smaller than the jackknife interval |
| 5 | common registration and ROI | **PASSES.** Node coordinates bit-identical across all 9 steps in both tests; `assert_common_roi()` enforces it |
| 6 | positive definiteness / conditioning | **PASSES.** `C_i` PD, cond 8.1e3 and 2.2e5. A Cholesky failure reports the spectrum rather than silently ridging |
| 7 | mixed-unit generalized kinematics | not exercised — `out_of_plane` means all three components are curvatures. Would need re-checking for `both` |
| 8 | validity of first-order perturbation | supported indirectly: the synthetic recovers `D_TRUE` to 0.1 % at the real noise level |
| 9 | data-dependent virtual fields | first-order plug-in, as derived. The synthetic bounds the residual bias at ≤ 0.1 % on `D11`/`D22` |
| 10 | cross-experiment independence | **ASSUMED.** Common systematic effects (DIC calibration, model error) are not represented and would not show up here |
| 11 | OLS vs FGLS | **FGLS better** on both independent cross-checks, and by 36–54 points on the synthetic |
| 12 | synthetic validation of the whole pipeline | **PASSES** — `workflows/validate_gls.py` test 3 |

Items 2, 7 and 10 are the ones still carried on assumption.

---

## Assumption audit

`python -m workflows.audit_assumptions` then `python -m workflows.make_audit_figures`. Every assumption behind
eqs. (1)–(20) was tested against the data rather than asserted; `figA_assumption_scorecard`
is the summary.

**Eq. (20) itself is not the problem.** A 150-replicate parametric bootstrap through the
whole pipeline — regenerating the fields from `Γ̂`, re-estimating `Γ` inside every replicate —
reproduces eq. (20) to a factor of 0.9–1.8 when the injected noise obeys eq. (2). The formula
is right. What is wrong is the noise model it is fed.

| eq. | assumption | verdict |
|---|---|---|
| §2–3 | `b` deterministic, noise perturbs `H` | **inverted, harmless.** The speciality condition pins `H = I` to 1e-15 for every realisation, so `δH = 0` and *all* the randomness sits in `b`. The algebra survives: `b_k = θ_k − Σ w (B_p u_k)ᵀK η_p`, so `δr` and hence `C` are the same object. Only the narrative in §2–3 needs correcting. |
| eq. (2) | noise spatially uncorrelated | **violated.** ρ = 0.92/0.94 at one grid step. **Not corrected** — eq. (3) is used verbatim; the violation is reported, not absorbed. See *No κ, and why*. |
| eq. (2) | `Γ` spatially homogeneous | **partial.** The correlation is anisotropic (0.73 along *x* vs 0.89 along *y*). |
| §8 | `M·Y ≈ 0` | **partial.** σ tracks the applied load (r = 0.88–0.99), so `Γ̂` is *effective* error, not pure DIC noise. Costs ≤ 1 % on θ. |
| §14 | first-order plug-in, `E*(θ)` from noisy data | **partial.** Bootstrap runs 1.2–3.5× wider than eq. (20) under eq. (2)'s own noise model. |
| §19 | eq. (20) treats `Γ̂`, `C` as fixed | **partial** — and the derivation says so itself. Jackknife/eq. (20) = 5.6, 1.8, 2.4, 0.7. |
| §1 | `rank(H_it) = p` ⇒ each step identifies all `p` | **vacuous.** `H = I` *by construction*, so this test always passes and can never detect that `D12` and `D66` carry no information. It is not a usable identifiability check. |
| eq. (12) | experiments independent | **untestable** with two coupons and no repeats. |
| §1 | one common `θ` fits all experiments | **violated.** The coupons disagree on `D66` by 7.8σ of their own error bars. |
| *implicit* | **no model error — the plate model is exact** | **violated, and dominant.** χ²/dof = 42. This assumption is never written down in the derivation, and it is the one that matters most. |

### What this means for the numbers

Applying κ would remove a factor of 20 from the misfit (χ²/dof 830 → 42) — but 42 would
still reject the model, and what survives is **not spatial correlation**. It is model error:
the Kirchhoff assumption, boundary treatment, and possibly unmeasured transverse shear (the
longitudinal README's Finding 3 puts it at ~17 %). Patching the covariance to absorb 20 of
the 830 would have hidden the diagnostic without fixing the cause.

Two consequences:

- **Every error bar must be inflated by ≥ 29×.** Eq. (20) is a repeatability statement, not
  an accuracy statement.
- **`D66` cannot be rescued by better statistics.** Its error is bias, and no variance
  formula — eq. (20), bootstrap or jackknife — can see bias. The tell is that `D66` is
  quoted to 9 % while sitting 11σ outside the physically feasible set.

`figB_uncertainty_ladder` shows the full ladder per coefficient; `figC_noise_model_fit` the
χ² test; `figD_spatial_whiteness` the measured correlogram against eq. (2);
`figE_D66_is_bias_not_noise` the `D66` case specifically.

---

## Figures

`python -m workflows.make_figures` writes `results/figures/` — vector PDF with Type-42 embedded fonts
(Type-3, matplotlib's default, is rejected by most journal submission systems) plus 600 dpi
PNG. Widths are the standard 3.5 in / 7.16 in column measures, so
`\includegraphics[width=\textwidth]{...}` needs no rescaling and the type comes out at the
intended size.

| figure | shows |
|---|---|
| `fig1_coefficient_evolution` | the four coefficients per iteration with the eq. (20) ±1σ band, plus a log-scale convergence panel |
| `fig2_correlation_determinant` | `log₁₀\|C_G\|` and `log₁₀\|R_G\|` per iteration |
| `fig3_parameter_variance` | σ per coefficient per iteration, absolute and relative |
| `fig4_information_pca` | principal std. deviations, eigenvector loadings, variance shares |
| `fig5_parameter_ellipsoid` | 4×4 corner plot: marginal and conditional slices through `(D11, D12, D22, D66)` |
| `figure_data.csv` | every number behind the figures |

Three things the figures say that the tables do not:

- **The stacked system is enormously redundant.** `log₁₀|R_G| = −40.1` for a 72×72
  correlation matrix — independent equations would give 0. This is the same fact as the
  0.64–0.77 cross-step correlation, read as one scalar.
- **The converged covariance is nearly diagonal.** Every principal component aligns with one
  `D` axis to ≥ 0.99, and `|corr| ≤ 0.06` throughout, so the marginal and conditional
  ellipses in fig. 5 coincide to 0.3 %. The four coefficients are determined essentially
  independently — fixing three of them buys almost nothing for the fourth.
- **`D22` dominates the uncertainty budget**, carrying 73.8 % of the total parameter
  variance against 5.2 % for `D11`.

Colour: one fixed hue per coefficient, never cycled, each also carrying a distinct marker so
identity never rests on colour alone. The set was checked with a CVD validator — worst
adjacent pair over *all* pairs is ΔE 9.2 (deuteranopia) and 16.3 (normal vision). Do not
swap the aqua for a green: green/orange collapses to ΔE 3.2 under protanopia.

---

## Files

| file | role |
|---|---|
| `vfm_constitutive.py` | the material matrices, and `rotate_constitutive_model` / `engineering_rotation_matrix` |
| `workflows/make_raw_frame_transverse.py` | undoes the export's +90° rotation to build route (B)'s input |
| `workflows/validate_rotation.py` | rotating `D` == rotating the fields, proved on the real coupon |
| `vfm_noise_covariance.py` | `Γ̂` from the residual projector; κ and ν; the `Φ` tensor; propagation to `C` |
| `vfm_gls_identification.py` | the FGLS loop, the per-iteration diagnostics, the jackknife |
| `workflows/campaign_common.py` | case building, orientation assertions, reporting |
| `workflows/run_campaign_gls.py` | the real run: noise diagnostics, comparison table, basis sweep, jackknife |
| `workflows/validate_gls.py` | the four-part known-answer benchmark |
| `workflows/make_figures.py` | the paper figures |
| `projects/campaign.vfmproj.json` | the loadable run: data files, meshes, BCs, settings |
| `vfm_project.py` | project format: save / load / run (no Qt) |
| `workflows/make_gui_project.py` | builds the project and proves it reproduces the campaign |
| `bc_gui/results_window.py` | the in-GUI results window |
| `bc_gui/experiment_setup_dialog.py` | the experiment / time-step setup dialog |
| `workflows/audit_assumptions.py` | the assumption audit: bootstrap, jackknife, per-experiment consistency |
| `workflows/make_audit_figures.py` | the audit figures |
| `results/audit/audit_results.npz`, `results/audit/audit_output.txt` | audit data and console log |
| `results/two_coupon/gls_results_orthotropic.csv` | the side-by-side table |
| `results/two_coupon/gls_step_estimates.csv` | per-load-step estimates entering the pooling |
| `results/two_coupon/campaign_full_output.txt` | full console output of `--all` |

Modified from the original solver, all small:

- `vfm_optimal_field.py` — `solve_optimal_virtual_fields()` assembles `G/A/R` once and solves
  all `m` right-hand sides together (2.8× faster; the single-parameter entry point is
  **bit-identical** to the original). `collect_assembly_payload()` made public.
- `vfm_user_inputs.py` — added `GlsSettings` and `UserInputs.experiment_ids`.
  `PARAMETER_LABELS_BY_MODEL` and `DOF_PER_NODE_BY_ANALYSIS` untouched.
- `bc_gui/run_setup_dialog.py` — one "Set experiment #…" button. The only GUI change.
- `run_vfm.py` — calls the GLS loop, reports the noise model.
- `tests/smoke_test.py` — the shadowing fix described above.

`run_identification_cases` is untouched and still reachable, which is what makes the OLS
baseline runnable for the comparison.

### Removed from the copy

The general-purpose solver shipped with assets this campaign does not use, and they are gone
(60 MB → 749 KB):

| removed | why |
|---|---|
| `Loading_Data/` | 37 demo `.mat` for other specimens (59 MB). This campaign reads `../Longitudinal_bending/` and `../Transverse_bending/`. The smoke tests build their own `.mat` in temp dirs, so nothing depended on it. |
| `doc/` | seven markdown files describing a workflow that no longer exists — they still documented an optional GUI and a `"Customize BCs? [y/N]"` prompt that the current `run_vfm.py` does not have. This README replaces them. |
| `boundary_conditions/*.py` | 34 generated BC files for unrelated specimens. **The directory stays** — it is where the GUI writes new BC files. |
| `VFM_bending_Pierron.m` | the MATLAB script the Python was ported from; provenance only, never executed. |
| `_membrane_subblock_check.py` | a standalone diagnostic for the membrane sub-block, not part of any run. |

`examples/boundary_conditions_plate_example.py` is **kept** — `tests/smoke_test.py` loads it.

One consequence needed fixing: `build_user_inputs()` seeded `data_files` from
`Loading_Data/`. It now seeds the nine longitudinal load steps with `experiment_ids = [1]*9`
and this campaign's mesh and initial guess, so `python run_vfm.py` opens ready to run. Add
the transverse steps as experiment 2 in the dialog to reproduce `workflows/run_campaign_gls.py` from
the GUI.

---

## The GUI

`python run_vfm.py` opens one dialog for the whole run, organised the way the method is:

```
EXPERIMENT   one physical test
             owns: boundary conditions, virtual mesh, sample area
  TIME STEP  one load step of that test
             owns: a .mat file, and the applied force/moment
```

You set **Number of experiments**, get one tab per experiment, and inside each tab you add
that experiment's time steps. Boundary conditions, the mesh, the **material frame
orientation** and the **applied-load source** are edited **once per experiment**; the applied
load itself is edited **per time step**, as a column in the step table — one column per
EXTERNAL group.

Whether those columns are live depends on the experiment's **Applied load from** selector
(see *Applied load* above): on `.mat` they are greyed and read-only with a note saying so; on
manual they are editable, blank cells are highlighted, and the run is refused until they are
filled.

The BC editor draws the **material 1- and 2-axes** on the canvas while you pick nodes,
anchored on the virtual mesh — or on the measurement cloud when no mesh has been applied yet,
which is how the editor opens from this dialog. The toolbar's **Material frame** toggle turns
them off; the angle lives in the Solver Settings tab and the arrows track it live.

### Why the split is exactly there

It is not a presentation choice, it is what the GLS requires:

- **BC topology is per experiment.** The optimal virtual field is built to annihilate
  specific unknown boundary terms. Change the constraint set between steps and the steps are
  no longer measuring the same quantity.
- **The virtual mesh is per experiment.** `C_{i,ts} = U_tᵀ G(Γ_ts) U_s` couples the virtual
  fields of steps *t* and *s*; two meshes with different DOF counts make that block
  undefined. (This is stricter than the request — the mesh cannot vary between time steps of
  one experiment. Two different meshes mean two different experiments.)
- **Only the load varies per step**, which is the whole reason for having several.

So the UI offers exactly that freedom and no more: an inconsistent run cannot be built by
accident. `check_experiment_consistency()` re-checks it before solving and names the
offending file if anything slipped through.

### Project files

Everything the dialog produces is a `VfmProject` — the same object a `.vfmproj.json` holds —
so a GUI run and a headless run are the same run. **Load project…** / **Save project…** are
in the dialog header.

`projects/campaign.vfmproj.json` ships with the folder: all 18 files, their experiment numbers, both
meshes, both BC sets, the model, the initial guess `[1,1,1,1]` and the solver settings.

```
python run_vfm.py                                            # GUI
python run_vfm.py --project projects/campaign.vfmproj.json            # headless + results window
python run_vfm.py --project projects/campaign.vfmproj.json --no-gui   # terminal only
```

`workflows/make_gui_project.py` rebuilds it from the same `TestSpec`s and shipped BC files the campaign
uses, then **proves** the claim rather than asserting it: it runs the identification through
both routes and refuses to leave the file on disk unless they agree. They agree to
**0.000e+00** — bit-identical.

### The results window

The run no longer ends in a terminal that scrolls away:

| tab | contents |
|---|---|
| **Coefficients** | value, eq. (20) σ, σ/\|value\|, and the ×√(χ²/dof) column — with the caveat text attached |
| **Evolution** | the four coefficients per iteration, plus a log-scale convergence panel |
| **Convergence** | per-iteration θ, ΔQ (eq. 18), ΔC (eq. 19), cond(whitened H), ‖H−I‖_F |
| **Noise model** | χ²/dof, per-experiment Γ, cross-step correlation, measured κ (flagged *not applied*), σ per load step |

**Virtual fields (3D)…** and **Identifiability…** jump straight on from there.

The window deliberately draws **no error band** on the evolution plot and shows both σ
columns side by side. When χ²/dof > 2 it displays a banner saying the noise model is rejected
and that eq. (20) is a variance *under the model*. A reader who takes only the headline away
should still not come away with a false precision.

---

## Headless API

The GUI is a front end for `VfmProject`; nothing is only reachable through it.

```python
from vfm_project import load_project, run_project
result = run_project(load_project(Path("projects/campaign.vfmproj.json")))
```

Or go straight to the solver with your own cases:

```python
run_gls_identification_cases(cases=..., experiment_ids=[1,1,1, 2,2,2], ...)
```

`experiment_ids` is what tells the GLS which files are load steps of one test. Files sharing
an id **must** share a virtual mesh, a BC topology and a measurement ROI — checked, not
assumed. Different ids are treated as statistically independent, which is what makes
`C_global` block diagonal.

---

## Caveats inherited from the source folders

The longitudinal test carries an unmeasured transverse shear (its README Finding 3, ~17 %
gap between its VFM and mean-field numbers), which the transverse test is largely free of.
The µCT thickness used in the derived `E_flex` readout was measured on the longitudinal
coupon only. The orientation argument that puts both datasets in one material frame is
verified numerically by `assert_orientation()` on every run.

---

# The 45° coupon — the off-axis test this folder was waiting for

`python -m workflows.run_campaign_45 --all` → the tables in `results/three_coupon/`;
the captured console log is `results/three_coupon/campaign_45_full_output.txt`.

Everything above this section is **unchanged**. `workflows/run_campaign_gls.py`,
`projects/campaign.vfmproj.json`,
`workflows/make_figures.py` and every published two-coupon number still reproduce bit-identically —
`workflows/validate_rotation.py` re-checks the campaign result to 1.05e-14 and re-confirms
`D11 = 174.3489`, `D22 = 61.4282`. `TESTS` was not touched; the new campaign is `TESTS_WITH_45`.

## The result

| | `D11` | `D12` | `D22` | **`D66`** | χ²/dof |
|---|---|---|---|---|---|
| 2 coupons *(the old campaign)* | 174.35 | 2.07 | 61.43 | **−3.66** | 830 |
| **3 coupons (with 45°)** | **153.31** | **−7.94** | **55.84** | **+9.87** | 2418 |
| eq. (20) σ, 2 coupons | 0.43 | 0.15 | 0.09 | 0.12 | |
| eq. (20) σ, 3 coupons | 0.31 | 0.12 | 0.08 | **0.03** | |
| jackknife σ, 3 coupons (27 replicates) | 6.60 | 3.50 | 2.56 | **1.45** | |

**`D66` changes sign.** The two-coupon campaign returned `D66 = −3.66`, which the README
already flagged as *"not a measurement — it is negative, which no real laminate can be"*.
With the 45° coupon it is **+9.87 N·mm**, and its eq. (20) σ drops by 4.5× while every other
σ moves by less than 30 %. That is the single thing this coupon was added to do.

## `D66 > 0` is robust; its magnitude is not

Every campaign containing the 45° coupon returns a positive `D66`; every campaign without it
returns a negative one:

| campaign | `D11` | `D12` | `D22` | `D66` | χ²/dof |
|---|---|---|---|---|---|
| all three | 153.31 | −7.94 | 55.84 | **+9.87** | 2418 |
| drop 45° *(the old campaign)* | 174.35 | 2.07 | 61.43 | **−3.66** | 830 |
| drop transverse | 178.24 | −1.02 | −35.60 | **+15.43** | 1100 |
| drop longitudinal | 69.01 | 4.73 | 57.62 | **+13.42** | 3574 |
| 45° alone | −14.48 | 28.58 | 58.34 | **+16.70** | 4517 |
| longitudinal alone | 181.70 | −0.02 | 1.26 | **−9.16** | 95 |
| transverse alone | 0.19 | −0.01 | 61.19 | **−2.01** | 1314 |

and the sign survives the smoothing-basis sweep (`D66` = 7.09 → 10.87 over degrees 2–10),
the 27-replicate jackknife (±1.45), the unknown specimen width, and the unknown cut sense
(see below).

> **It does NOT survive the boundary-condition form.** The BC study in the last section of
> this README finds `D66` from −2.20 to +20.58 across admissible forms, negative exactly when
> the free-edge condition is not used. Read that section before quoting a sign. The magnitude
> is softer still — see the cross-check two sections down.

## Where the D66 information comes from — read this before the numbers

At exactly 45° the `D66` basis matrix rotates onto the **normal** curvatures:

```
T(45)ᵀ [[0,0,0],[0,0,0],[0,0,1]] T(45)  =  [[ 1,-1, 0],
                                            [-1, 1, 0],
                                            [ 0, 0, 0]]
```

so `D_lab[0,0] = D_lab[1,1] = S + D66` and `D_lab[0,1] = S − D66` with
`S = (D11 + D22 + 2·D12)/4` — Jones's closed forms, which
`rotate_constitutive_model()` already asserts. With the lateral edges traction-free
(`M_xx = 0`, which is exactly what `BC_FortyFive.py` buys by leaving them unconstrained):

```
κ_xx / κ_yy  =  −(S − D66) / (S + D66)
```

**The anticlastic ratio *is* the `D66` measurement.** Measured on this coupon:
**−0.3814 ± 0.0173** over nine load steps, stable to 2 % across the whole trim sweep. The
same ratio is −0.03 on the longitudinal coupon and −0.01 on the transverse one, where it is
noise — which is precisely why neither carries any `D66` information, and why the README's
`H = I` identifiability test could never see the difference.

This route matters because **the fixture restrains the twist**. A 45° coupon in bending wants
to twist, but the yoke grips both ends and holds it: `ΔΘ_x` stays within 0.04° over the whole
ramp, and the measured `κ_xy/κ_yy` is **−0.022** against the ≈ −0.33 a free coupon would
show — a factor of 15 of suppression, with an unmeasured reaction twisting moment appearing
at the cuts instead. A `D66` route that needed a measurable twist would fail here. This one
does not need one.

## The cross-check that does not agree, and why it is the interesting one

The closed form above is an **independent** route to `D66`, using only the two normal
curvature channels of the 45° coupon plus the `D11, D12, D22` the other two supply:

```
measured r                      = −0.3814 ± 0.0173   (9 load steps)
S from the fitted D11, D12, D22 =    48.32 N·mm
D66 from the closed form        =    21.64 N·mm
D66 from the full FGLS          =     9.87 N·mm      gap 54 %
```

They disagree by more than a factor of two, and the reason is worth stating plainly: the 45°
coupon supplies **two** observables, and they are mutually inconsistent given `S`.

* the **ratio** `κ_xx/κ_yy` — a ratio of two curvature fields measured at the same points in
  the same frame, so the moment, the specimen width, the machine calibration and the moment
  gradient all **cancel out of it exactly**;
* the **magnitude** `M_yy/κ_yy ≈ 47 N·mm` — which depends on every one of them.

With `S = 48.32`, the fitted `D66 = 9.87` predicts `r = −0.661` against the measured −0.381,
while `D66 = 21.64` predicts an effective stiffness of 59.8 N·mm against the measured 47.
The FGLS splits the difference, and χ²/dof = 2418 is it reporting that it cannot do both.

The noise covariance cannot know that one observable is clean and the other is contaminated
by model error, so it weights them by measurement noise alone. **The ratio is the observable
to trust**, for the reason above. Read `D66` as bracketed by the two routes:

```
D66 = 10 to 22 N.mm,   positive,   with the ratio-only estimate at the top of that range
```

Not a 9 %-precision number. But it is the first `D66` from this campaign that is not
physically impossible.

## Two open inputs, and an envelope instead of a σ

Neither is a statistical uncertainty — no covariance can see either — so they are swept
rather than propagated (`--open-inputs`).

**(1) The specimen width was never measured for this coupon.** 9.28 mm is a placeholder; the
DIC field spans 9.00 mm and the two measured coupons ran 1.007× and 1.038× their own DIC
span, bracketing 9.06–9.34 mm. Every moment scales as `1/width`. It turns out to matter
little: over 9.00–9.60 mm, `D11` moves 2.1 %, `D22` 0.09 %, `D66` 11.6 %.

**(2) The cut sense — is the material 1-axis at +45° or +135°?** The coupon length runs along
its own +y, and the 1-axis is 45° from the length, so it sits at `90 − 45 = 45°` or
`90 + 45 = 135°`. Both are geometrically possible and the specimen drawing carries only
nominal sizes. Everything **even** in the angle is identical between them — `T(135)` differs
from `T(45)` only in the sign of its third row and column, so the effective bending stiffness,
the anticlastic ratio and hence `D66` cannot distinguish them. Only the sign of the
bend-twist coupling can, and that is the channel the fixture suppressed.

The measured `κ_xy/κ_yy` is negative at **all nine** load steps (−0.009 to −0.047), which is
the sign `T(45)` predicts for `D66 > 0`; that is the shipped choice. It is **weak** evidence,
and χ²/dof does not settle it either (2418 for 45° against 2355 for 135°).

Swept together, over the four corners of the plausible box:

| | `D11` | `D12` | `D22` | `D66` |
|---|---|---|---|---|
| envelope min | 124.68 | −8.01 | **55.84** | **+9.75** |
| envelope max | 154.52 | −6.20 | **59.13** | **+11.14** |
| span, % of midpoint | 21.4 % | 25.5 % | **5.7 %** | **13.3 %** |

**`D66` stays positive over the whole envelope**, and `D22` moves by 5.7 %. `D11` moves 21 %
— but `D11` was already the coefficient this campaign cannot pin down, with a **146 %** spread
from the boundary-condition form alone on the longitudinal coupon. The cut-sense ambiguity is
therefore *smaller than the uncertainty `D11` already carried*, and does not touch the
coefficient the new coupon was added for.

**To close it:** photograph or measure which diagonal of the plate the coupon was cut along.
One fact, and the envelope collapses to one column.

## What got worse: χ²/dof 830 → 2418

Adding a third coupon nearly tripled the misfit. That is not a defect of the estimator, it is
the three coupons disagreeing about one `D`, and it is worth being precise about where.

The 45° coupon's own effective bending stiffness is `M_yy/κ_yy ≈ 47 N·mm`. Combined with its
anticlastic ratio that implies `S ≈ 38` and hence `D11 + 2·D12 ≈ 91` — against the
longitudinal coupon's `D11 ≈ 174` (VFM) or `203` (its own single-test fit). This is the same
tension the two-coupon campaign already had in `D11`, now visible from a third direction.

The likely cause is the 45° coupon's own weakness, and it is documented rather than hidden:
**`κ_yy` reverses sign at `y ≈ 4.2 mm` on this specimen.** The deflection increment has an
interior maximum near `y ≈ 10 mm`, confirmed from raw central differences on `W` independently
of Vic-3D's curvature filter (`../45deg_bending/NOTES_45deg_frame_and_curvature.md` §5). The
VFM window starts at `y = 5.30 mm` to stay clear of the inflection; what remains is a **107 %**
end-to-end gradient in the dominant curvature, against 166 % longitudinally and 23 %
transversally. `F = [−M, +M]` at both cuts assumes the two cut moments are equal, and a 107 %
gradient is how far from true that is.

**This is why the ratio survives and the magnitude does not**, and it is why the cause of the
inflection — most likely a different grip position or free span for this specimen, or the DIC
window straddling a support — should be resolved from the fixture photographs before this
coupon's `S` contribution is quoted.

## The boundary conditions, and why they are the longitudinal file verbatim

`45deg_bending/VFM_export/BC_FortyFive.py` is a character-for-character copy of the clamp
structure in `BC_Sample_6.py`. That is deliberate: **geometrically this coupon sits exactly
where the longitudinal one sits** — length along +y, width along +x, loaded cuts top and
bottom, free edges left and right. The BCs describe the *fixture*, which is the same fixture.
What differs is the *material frame*, and that is the solver's job, not the BC file's.

| | top / bottom cut | left / right |
|---|---|---|
| `W`, `W_X` (tangential slope), `W_XY` | **= 0** | **untouched** |
| `W_Y` (normal rotation) | **free** → does external work | **untouched** |
| EXTERNAL | `M_yy`, distributed, sort `x` | none |

Two clamps carry more weight here than on the other two coupons:

* **Leaving `left`/`right` free is the mechanism, not a convenience.** `M_xx = 0` on the free
  edges is exactly the condition that turns the measured anticlastic ratio into a `D66`
  measurement. This window also reaches those edges better than either other test — 0.14 mm
  short per side (**1.5 %** of the width) against 3.0 % longitudinally and 6.4 % transversally.
* **`W_XY = 0` on the cuts annihilates a real, unmeasured reaction.** Both axis-aligned coupons
  have `D16 = D26 = 0` in their own frame, so their cuts carry essentially no twisting moment
  and that clamp costs nothing. This coupon has `D'16 = D'26 = (D11 − D22)/4 ≈ +24 N·mm`: it
  genuinely wants to twist, the yoke holds it, and the resulting reaction is unmeasured. That
  clamp is what keeps it out of the work balance.

`BC_BOTH_CUTS` (form (a)) is used, for the same reason the longitudinal coupon uses it: on
this rig *neither individual cut moment is known*, the machine reports a single couple, and
with a virtual field antisymmetric along the length form (a)'s external work depends on the
**mean** of the two — which the machine couple does estimate well. Form (b) depends on a
single cut moment, which it does not.

## Nothing was rotated except `D`

This coupon is route (B) and **only** route (B) — there is no pre-rotated twin and there
cannot be one, because a 45° rotation *mixes* `κ_xx`, `κ_yy` and `κ_xy` instead of permuting
them, so the `cxy` factor-of-two and sign would contaminate all three channels. The `.h5` are
in the specimen frame, the `.mat` are in the specimen frame, the mesh and the wrench are in
the specimen frame, and `material_orientation_deg = 45` tells the solver to rotate `D`.

This is exactly the code path the *Frames: rotate `D`, not the fields* section was built for.
The README's closing line there — *"A 45° coupon needs nothing further — it is this code path
with a different angle"* — held: no solver change was needed, only a new `TestSpec`.

The curvature convention was **re-measured on this dataset** rather than inherited: raw central
differences on `W` give `∂²W/∂Y² / (2·Δcyy) = +1.02`, `∂²W/∂X² / (2·Δcxx) = +1.05`,
`∂²W/∂X∂Y / Δcxy = −1.13`, and the Kirchhoff surface offset `z = −ey/κ_yy = +0.130 mm`
against the µCT half-thickness 0.1282 mm. **The twist sign matters here**, unlike on the other
two coupons where `κ_xy` carries no information, so it is measured, not assumed.

## The orientation check had to be relaxed — carefully

`assert_orientation()` required the length-conjugate curvature to exceed the other by **5×**.
An off-axis coupon fails that legitimately: at 45° the anticlastic ratio is `−(S−D66)/(S+D66)`,
which for any physical `S, D66 > 0` lies strictly inside (−1, 1) and here measures −0.38, so
the dominance is ≈ 2.6× **by construction**.

Lowering the threshold alone would have removed a real guard, so `TestSpec` gained **two**
fields and the strictness moved rather than vanished:

* `min_curvature_dominance` — 5.0 for the axis-aligned coupons, 2.0 for this one;
* `max_anticlastic_ratio` — `+inf` (vacuous) for the axis-aligned coupons, **1.0** for this
  one, where `|κ_other/κ_dominant| < 1` is exactly the statement that `S` and `D66` are both
  positive. It is a **positive-definiteness check on `D`, made against the measured fields**,
  and it would fire on a wrong frame, a wrong moment sign or a wrong curvature convention.

```
45deg   span x 8.87 mm, y 9.27 mm  -> length along y                          [OK]
        mean kappa_xx -9.536e-03, kappa_yy +2.713e-02
          -> dominant |yy| is 2.8x the other (need > 2.0x)                    [OK]
        anticlastic ratio -0.3514
          -> |.| < 1.0 required for S, D66 > 0 at 45 deg                      [OK]
```

## Files added

| file | role |
|---|---|
| `../45deg_bending/VFM_export/export_vfm_dataset.py` | `.h5` → 9 `.mat`, specimen frame, no rotation; `--trim-sweep` |
| `../45deg_bending/VFM_export/BC_FortyFive.py` | the boundary conditions |
| `../45deg_bending/VFM_export/mat/FortyFive_step{03..11}.mat` | 1978 points each, 43 × 46 grid |
| `../45deg_bending/VFM_export/vfm_export_manifest.csv` | per-step moment, curvatures, anticlastic ratio, gradient |
| `../45deg_bending/VFM_export/roi_geometry.csv` | window geometry + outward-rounded mesh bounds |
| `workflows/run_campaign_45.py` | the three-coupon driver |
| `results/three_coupon/campaign_45_full_output.txt` | full console output of `--all` |
| `results/three_coupon/gls_results_orthotropic_with45.csv`, `results/three_coupon/gls_step_estimates_with45.csv` | results |

Changed in `workflows/campaign_common.py`, all additive: `FORTYFIVE_DIR`, the `FORTYFIVE` `TestSpec`,
`TESTS_WITH_45`, and the two new `TestSpec` fields with their check in `assert_orientation()`.
`TESTS` is unchanged, so `workflows/run_campaign_gls.py` and everything downstream is untouched.

## Verification

| check | result |
|---|---|
| `workflows/validate_rotation.py` — full two-coupon campaign, route A vs B | 1.05e-14, **ALL CHECKS PASSED** |
| two-coupon campaign result vs published | `D11` 174.3489, `D22` 61.4282, drift ≤ 0.0018 |
| `workflows/validate_gls.py` — 13 known-answer checks | **ALL CHECKS PASSED** |
| `tests/smoke_test.py` | **all checks PASSED** |
| 45° orientation check (geometry, dominance, positive definiteness) | all OK |
| every measurement point maps into the virtual mesh | 0 unmapped, all 9 steps |
| `C_i` positive definite, 45° coupon | PD = True, cond = 3.89e+05 |
| FGLS convergence, three coupons | 7 iterations, ΔQ = 4.2e-11 |

## What to quote

**`D66 = +9.9 N·mm` from the full FGLS, or 10–22 N·mm bracketing the ratio-only route —
positive either way, and positive across every sensitivity run.** That is the result.

`D11` and `D22` should still be quoted from the two-coupon campaign unless the cut sense is
settled: adding the 45° coupon moves them (174 → 153, 61 → 56) but it moves them using a
coupon whose own moment magnitude is compromised by a 107 % gradient, and it inflates
χ²/dof from 830 to 2418. The new coupon's job was `D66`, and `D66` is what it delivered.

---

# The 45° coupon: boundary conditions, window trim, and what D12 is doing

`python -m workflows.bc_study_45` writes the CSV to
`results/bc_studies/three_coupon/` and figures to `results/figures/study_45/`;
the captured console log is stored beside the CSV. Stacked three-coupon versions of the five
campaign figures: `python -m workflows.make_figures --with-45 --bc-form both-cuts` →
`results/figures/with_45/both_cuts/`.

The longitudinal and transverse coupons are held at their shipped reference in every run
below, so every difference belongs to the 45° coupon.

## The trim objection was right, and the shipped fix was the wrong one

The objection: *if the virtual field annihilates every unknown traction on the window
boundary, the VFM identity holds on any subdomain, so the answer must not depend on where the
cuts are placed.* That is correct, and the trim sweep shows exactly where the shipped setup
fails it.

Spread of each coefficient as the window start is swept from `y = 1.07 mm` (bare 15-filter
trim — the window **straddles** the curvature reversal at `y ≈ 4.2 mm`) to `y = 6.31 mm`:

| form | what it asserts | `D11` | `D12` | `D22` | `D66` |
|---|---|---|---|---|---|
| **A** both cuts loaded *(shipped)* | machine couple at **both** cuts | 21 % | 142 % | 3.4 % | 56 % |
| **B** top loaded, bottom **sealed** | machine couple at the **top** cut only | **3.0 %** | **5.7 %** | **3.0 %** | 34 % |
| **C** bottom loaded, top sealed | machine couple at the **bottom** cut only | 26 % | 270 % | 5.6 % | 2349 % |
| **D** both cuts loaded + linked `W_Y` | as A, smaller field space | 5.7 % | 135 % | 2.1 % | 14 % |
| **E** bottom `W_XY` free *(leaks)* | as B, plus a leak | **1.7 %** | 11 % | 2.8 % | 28 % |
| **F** top `W_XY` free *(leaks)* | as B, plus a leak | 3.8 % | **4.6 %** | 2.9 % | 35 % |
| **G** lateral edges `w* = 0` | as A, free edges partly used | 2.0 % | 38 % | 2.5 % | 646 % |
| **H** lateral edges fully sealed | free-edge condition **not used** | **1.7 %** | 995 % | **1.3 %** | 182 % |
| **S** both cuts sealed *(scale-free)* | **nothing** | — | 98 % | 20 % | **17 %** |

**The trim dependence is not in the fields. It is in which cut the machine couple is asserted
at.** The low-`y` cut is the one that marches into the curvature reversal as the trim is
relaxed. Form A feeds the couple into *both* cuts, so it inherits that error in full and
drifts 21 % on `D11`. Form B **seals the bottom cut** — asserting nothing about it — and the
top cut never moves, so B is flat to 3 % *including the window that straddles the reversal*.
Form C loads the bad cut and seals the good one, and is catastrophic.

So the heavy trim was treating a symptom. The window does not need to be cut back to
`y = 5.3 mm`; the bottom cut needs to be sealed. **`LOW_Y_TRIM_STEPS = 28` in the shipped
export can be relaxed to 7 if form B is used**, recovering 2881 points instead of 1978.

### But B's premise is the one this rig actually violates

Form B asserts `M_yy(top) = M_machine`. The README's *Why the moment is applied at both ends*
section measures the violation: an unmeasured dead weight (~170 g of grip and fixture outboard
of the load cell) makes the two cut moments differ by ~100 % of the applied couple, so **no
individual cut moment is known on this rig** — only, approximately, their mean. That is why
form A exists at all, and why the longitudinal coupon uses it.

So A and B trade one error for the other:

```
A  right about the moment (mean), wrong about the gradient  ->  D66 = 9.87
B  wrong about the moment (one cut), right about the gradient -> D66 = 3.63
```

They bracket. Neither is a refutation of the other, and the campaign ships A for consistency
with the other two coupons — but B is the form to prefer if the dead weight is ever measured
and removed.

## The scale-free form — the objection's ideal case, built and run

Seal **both** cuts (`W`, `W_X`, `W_XY`, `W_Y` all prescribed) and no traction, known or
unknown, does any virtual work. The external work is identically zero and the VFM reduces to
the homogeneous system

```
(Rᵀ A) θ = 0            A[:,j] = Σ_p w_p B_pᵀ B_j^lab κ_p ,  R = admissible basis
```

**No moment, no specimen width, no machine calibration and no moment gradient enters** — every
quantity the objection is right to distrust drops out. The price is that only the *direction*
of θ is fixed; scaling `D` scales both sides.

> Note this cannot be done through the optimal-virtual-field route: the speciality condition
> pins `H = I` by construction, so a zero right-hand side would simply return `θ = 0`. The
> homogeneous system has to be assembled directly, which `sealed_design_matrix()` does.

Result, normalised on `D11`, across the whole trim sweep:

```
window start y   1.07   2.07   3.08   4.09   5.30   6.31  mm
D66 (scaled)    22.31  24.32  20.58  21.78  21.71  21.05        spread 17 %
```

**Stable to 17 % from a window straddling the curvature reversal to the shipped one** — the
objection's prediction, confirmed. And it agrees with the closed-form anticlastic-ratio route
(`D66 = S(1+r)/(1−r) = 21.6`) to 1 %, which is a genuinely independent check: both are
moment-free, but one uses only the mean curvatures and the other the full field against a
12-dimensional virtual-field space.

**The caveat is real, though.** The null direction is only weakly determined — singular gap
`σ₃/σ₄ = 1.6`. That is not a numerical accident: a 45° coupon constrains only **two** of the
four parameters (`S` and `D66`, see below), so two directions are genuinely free and the SVD
has to pick among near-degenerate ones. The sealed form therefore cannot stand alone; it is a
consistency check on the loaded forms, and a strong one.

## Correction: `D66 > 0` is NOT robust to the boundary-condition form

An earlier summary of this work said `D66` stays positive across every sensitivity. That was
true of the sensitivities tested then (cut sense, specimen width, smoothing-basis degree,
jackknife) but **it does not survive the BC study**, and the exception is systematic rather
than random:

| form | uses the free-edge condition `M_xx = 0`? | `D66` |
|---|---|---|
| A both cuts loaded | yes, fully | **+9.87** |
| B top loaded | yes, fully | **+3.63** |
| C bottom loaded | yes, fully | **+20.58** |
| D linked `W_Y` | yes, fully | **+8.99** |
| E, F leaks | yes, fully | +3.57, +6.34 |
| G lateral `w* = 0` | partly (`W_X` still free) | **+3.42** |
| **H lateral edges fully sealed** | **no** | **−2.20** |

This is exactly what the mechanism predicts. `D66` is carried by the anticlastic ratio
`κ_xx/κ_yy = −(S−D66)/(S+D66)`, and that relation *is* the statement `M_xx = 0` on the lateral
edges. Seal those edges and the information is gone — H simply reverts to the two-coupon
answer (`D66 = −2.2` against the two-coupon `−3.66`).

So the honest statement is:

> **`D66 = +3 to +21 N·mm`, positive in every form that uses the free-edge condition and
> negative in the one that does not.** The result rests on the free edges being genuinely
> traction-free — which this coupon supports better than either other test (window 0.14 mm
> short per side, 1.5 % of the width, against 3.0 % and 6.4 %), but which the BC study
> assumes rather than proves. The remaining check is a lateral-trim sweep of the kind
> `explore_free_edges.py` runs on the transverse coupon.

`D66` is still the coefficient this coupon was added for, and it is still the first evidence
in this campaign that `D66` is positive. It is not a 9 %-precision number.

## Why `D12` is the most sensitive coefficient

At 45° the lab-frame basis matrices, normal block only:

| | `[0,0]` | `[0,1]` | `[1,1]` |
|---|---|---|---|
| `D11` | 0.25 | 0.25 | 0.25 |
| `D12` | 0.50 | 0.50 | 0.50 |
| `D22` | 0.25 | 0.25 | 0.25 |
| `D66` | 1.00 | **−1.00** | 1.00 |

`D11`, `D12` and `D22` produce the **same pattern**, differing only in weight. So in its
normal channels a 45° coupon sees exactly **two** numbers:

```
S = (D11 + D22 + 2 D12) / 4        and, separately,        D66
```

It cannot separate `D11`, `D12` and `D22` at all — that separation comes from the two
axis-aligned coupons. Inverting:

```
D12 = 2 S − (D11 + D22)/2
```

which at the shipped point is `96.64 − 104.58 = −7.94`: **a difference of two quantities of
order 100 giving a result of order 8, a 13-fold cancellation.** A 1 % error in either term is
a 13 % error in `D12`.

Measured across the eight loaded BC forms — the identity reproduces every row to 7.1e-15:

| | `D11` | `D22` | `(D11+D22)/2` | `S` | `2S` | `D12` |
|---|---|---|---|---|---|---|
| spread across forms | 29 % | **7.8 %** | 22 % | **37 %** | 37 % | **481 %** |

**`S` moves 37 % while `D12` moves 481 %.** The identification is determining `S` perfectly
respectably; `D12` is a *derived* difference, and the derivation is what amplifies. `D22`, by
contrast, moves only 7.8 % and is the best-determined coefficient in the whole campaign.

Two further points:

* `S` is the 45° coupon's **magnitude** observable — the one the moment gradient and the
  specimen width contaminate — whereas its **ratio** observable carries `D66`. So `D12`
  inherits twice the error of the least trustworthy thing this coupon measures, plus half the
  error of `D11`, which the campaign already could not pin down.
* **This is not a 45°-specific defect.** `D12` was already unresolved on two coupons: 0.44
  (top-only) to 2.07 (both-cuts), with a Birge interval of ±4.33 — an interval twice its own
  value, consistent with zero. The 45° coupon does not make `D12` worse; it makes the reason
  *visible*, because it is the first test that constrains the combination `D12` sits inside
  at all. Before it, `D12` was free and drifted to whatever the noise preferred.

## Figures

| figure | shows |
|---|---|
| `results/figures/study_45/fig45_bc_forms` | the four coefficients under all eight loaded BC forms, against the two-coupon campaign |
| `results/figures/study_45/fig45_bc_forms_key.txt` | what each variant constrains, and what it assumes |
| `results/figures/study_45/fig45_trim_sweep` | coefficients vs window start, one panel per BC form — the trim answer |
| `results/figures/study_45/fig45_sealed_ratios` | the scale-free form's ratios vs trim — moment-free, and trim-free |
| `results/figures/with_45/both_cuts/fig1…fig5` | the five campaign figures for the stacked three-coupon run |

## What to change in the shipped setup

1. **Relax the trim if form B is adopted.** `LOW_Y_TRIM_STEPS` can go from 28 back to 7,
   recovering 2881 points, *provided* the bottom cut is sealed. Under form A the trim is
   load-bearing and 28 is the right value.
2. **Measure the fixture dead weight.** It is what forces the choice between A and B, and it
   is the single measurement that would collapse the `D66 = 3.6 vs 9.9` ambiguity.
3. **Run a lateral-trim sweep** to test the free-edge condition directly, since the whole
   `D66` result rests on it (variant H shows what happens without it).
