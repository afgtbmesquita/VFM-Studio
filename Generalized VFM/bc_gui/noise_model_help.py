"""Explanation dialog for the noise model / weighting controls.

Mirrors ``Generalized_VFM_GUI/src/vfmstudio/gui/dialogs/noise_model_help.py``
-- the two front ends (abdXtract and this run-setup dialog) are independent
trees by design (see the "Generalized VFM" / bc_gui README), so this is a
deliberate copy, not an import.  Keep both in step with each other and with
``GLS vs OLS/README.md`` in the workspace root, the source of the numbers
quoted here.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QTextBrowser,
    QVBoxLayout,
)

NOISE_MODEL_HELP_HTML = """
<h2>The noise model, Gamma</h2>

<p>FGLS is iterative because the weights depend on the answer:</p>
<pre>
repeat:
    D(theta)          build the stiffness matrix from the current theta
    U_t = solve(D)    optimal virtual fields, per load step
    C   = f(Gamma, D, U)   the equation-error covariance
    theta = weighted least squares with C
until theta settles AND C settles
</pre>

<p>Two structural facts make everything below simple. First, this solver's
speciality condition makes the per-step design matrix the identity, so
<b>every load step already gives a complete, standalone estimate of every
parameter</b>. Combining several load steps, or several tests, is therefore
not really a regression -- it is <i>"I have several estimates of the same
vector, how do I combine them?"</i> Second, Gamma and the virtual fields are
separate objects: Gamma is a statistical model of the <b>measurement noise</b>;
the virtual fields are the mechanics. Gamma never touches the mechanics -- it
only sets the weights used to combine the estimates.</p>

<h3>How Gamma is constructed</h3>

<p>You cannot observe noise directly, only the measured field. Gamma is built
by declaring the <b>true</b> field smooth and calling whatever is left over
"noise":</p>
<pre>
R_t = (I - Q Q') Y_t        Q = an orthonormal 2-D polynomial basis
Gamma = Z' Z / nu           Z = the stacked residuals R_t across load steps
</pre>

<p>The polynomial's <b>total degree</b> is the "Gamma basis degree" spin box.
Degree 6 gives d = (6+1)(6+2)/2 = <b>28</b> basis functions (the "d = 28" the
solver prints). This is the only setting in the whole panel that is not part
of the VFM itself -- it is purely a modelling choice about what counts as
noise.</p>

<p><b>The trade-off is sharp in both directions.</b> Too low a degree, and
real mechanics that the polynomial cannot represent gets called noise: Gamma
is inflated and mis-shaped in exactly the directions the physics lives. Too
high, and the polynomial starts fitting the noise itself, so Gamma is
underestimated. On this campaign's topology-optimised coupons the kappa_xy
field around the design cutouts -- the entire reason those specimens exist --
is not a degree-6 polynomial, so part of it lands in Gamma. Sweeping the
degree from 6 to 16 on one such coupon moved the identified D26 from 8.2 to
12.7 N.mm. <b>Sweep the degree; if the parameters move, the noise model is
carrying your result, not the data.</b></p>

<h3>Spatial correction (kappa)</h3>

<p>Off by default. The default treats every DIC facet as an independent
sample. That is false -- Vic-3D's curvature filter correlates neighbouring
facets -- so turning this on <b>measures</b> the number of effectively
correlated points, kappa, from the residual field and applies it two ways:
the normaliser becomes <code>nu = N - kappa*d</code> instead of
<code>N - d</code> (a small effect), and <code>C</code> is scaled by
<code>kappa</code> (the large one).</p>

<p>Measured on this campaign: kappa is roughly 4-11 on the solid rectangular
coupons and 24-36 on the topology-optimised ones (their residual is less
polynomial, for the same reason as above).</p>

<p>Because kappa multiplies the WHOLE of C by one number, it is a uniform
scalar and <b>cancels out of theta exactly</b> for a single test -- turning
this on changes the identified parameters by about 1e-14, i.e. not at all.
What it changes is how honestly the fit is reported: chi-squared/dof divides
by roughly kappa, the model-based sigma multiplies by roughly sqrt(kappa),
and the Birge-scaled sigma (sigma times sqrt(reduced chi-squared)) is
<b>exactly invariant</b>, because the two changes cancel by construction.
Across a stack of several EXPERIMENTS the per-experiment kappa can differ, so
turning this on there does reweight the experiments against each other and
can move theta slightly -- but only because different experiments turned out
to have different noise levels, not because the correction is "more
correct" in one direction.</p>

<h3>Weighting -- and why there are three controls, not one</h3>

<p>Once Gamma exists, it is turned into the equation-error covariance
<code>C_ts = kappa * U_t' G(Gamma_ts) U_s</code>, and theta is the
generalised-least-squares combination of the per-step/per-test estimates
using C. Four choices for how much of C to use are offered:</p>
<ul>
<li><b>gls</b> -- the full dense block, cross-time/cross-test correlation kept.</li>
<li><b>block_diagonal</b> -- only the diagonal blocks (own step/test only).</li>
<li><b>diagonal</b> -- also drop cross-parameter terms within a block.</li>
<li><b>ols</b> -- C = the identity, i.e. plain least squares, every estimate
weighted equally.</li>
</ul>

<p>The single "Weighting" box applies <b>one</b> of these rules to
<i>both</i> jobs at once: combining the load steps of one test, and combining
separate tests. Those are not the same job. Load steps of one test are
repeat measurements of the <b>same</b> specimen -- same speckle pattern, same
reference frame, same systematic errors -- and are typically 0.9-0.999
correlated. Giving that a full "gls" weighting does not average them, it
<b>differences</b> them: the weights can run from -1 to +1 across the steps,
and the combined answer can land <i>outside</i> the range of what any single
load step measured. That is the mathematically optimal thing to do only if C
is exact, and here C is estimated from an effective rank of only a handful
out of dozens of rows -- built from the very data it is weighting. Separate
tests, by contrast, are genuinely independent and carry different
information (different orientation, different loading mode), so weighting
<b>them</b> by their covariance is ordinary, well-founded inverse-variance
pooling: plain averaging (ols) across tests is what goes badly wrong, because
a test that knows nothing about a parameter still votes on it with full
weight.</p>

<p>The <b>Load-step estimator</b> and <b>Experiment estimator</b> boxes
split the single Weighting rule into these two separate jobs. Left at "use
Weighting above" on both, the panel is in <b>legacy, single-axis mode</b> and
reproduces the historical behaviour exactly. Setting either one switches to
the explicit <b>two-stage</b> estimator: stage 1 combines each experiment's
own load steps into one estimate (and its own honest covariance); stage 2
combines the per-experiment estimates. Once either axis is set, the single
"Weighting" box is disabled and ignored -- the two-stage machinery always
builds the full covariance internally and lets the two axes decide how much
of it to use at each stage.</p>

<p>The recommended pairing, and the one this workspace's own analysis
(<code>GLS vs OLS/README.md</code>) argues for, is
<b>Load-step = ols, Experiment = gls</b>: average the repeat measurements
of one specimen (removing the differencing risk), then weight the
independent specimens by their measured precision (keeping the one place
GLS weighting is unambiguously well-founded).</p>

<h3>When is Gamma actually doing anything?</h3>

<p>Gamma reaches theta only through C, so several settings make the basis
degree (and the spatial correction) completely inert -- the panel disables
them and says so when this happens:</p>
<ul>
<li><b>Weighting = ols</b> (legacy, single-axis): C = the identity, Gamma is
never even read.</li>
<li><b>Load-step = ols, one experiment only</b>: the load steps are averaged
and there is nothing left to combine, so Gamma has no effect on theta.</li>
<li><b>Load-step = ols, Experiment = ols</b>: every experiment is weighted
equally, so Gamma drops out of theta entirely (though it still, harmlessly,
shapes the reported +/- under Load-step = gls/diagonal/block_diagonal).</li>
</ul>

<p>One case is easy to get backwards: with <b>Load-step = ols and several
experiments</b>, Gamma no longer sets the within-test weights -- but it
still sets each experiment's own covariance, which is exactly what decides
how much that experiment counts relative to the others. Removing Gamma's
first job concentrates <i>all</i> of its remaining influence into the
second one, so the basis degree matters <b>more</b>, not less, in this
configuration. Measured on a 4-coupon stack: sweeping the degree from 6 to
12 moved theta by up to 7.8 under full gls/gls weighting, but by up to
<b>14.1</b> under Load-step = ols, Experiment = gls. Always sweep the degree
under whatever pairing you actually intend to quote.</p>

<h3>Convergence settings are a different question</h3>

<p><b>Max iterations</b>, <b>Parameter tolerance</b> and
<b>Covariance tolerance</b> (grouped separately, below the noise model) do
not change which answer the solver converges to -- only whether it is given
enough iterations to get there, and how strictly "settled" is defined. The
loop stops only once both the parameters and the re-estimated covariance
have stopped moving.</p>
"""


class NoiseModelHelpDialog(QDialog):
    """Read-only explanation of the noise model / weighting panel."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Noise model and weighting -- how it works")
        self.resize(720, 720)
        layout = QVBoxLayout(self)
        browser = QTextBrowser(self)
        browser.setOpenExternalLinks(True)
        browser.setHtml(NOISE_MODEL_HELP_HTML)
        layout.addWidget(browser)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(
            self.accept
        )
        layout.addWidget(buttons)


def show_noise_model_help(parent=None) -> None:
    """Open the noise-model help dialog.

    Modal by design: it is read-only documentation, not a tool a user needs
    open side by side with editing the panel, and exec() sidesteps any
    lifetime/garbage-collection bookkeeping for a fire-and-forget dialog.
    """
    NoiseModelHelpDialog(parent).exec()
