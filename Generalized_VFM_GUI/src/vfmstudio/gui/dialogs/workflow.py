"""One-page modelling workflow shown from Help."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout

from vfmstudio.branding import APP_NAME

WORKFLOW_TEXT = """
<h3>Typical workflow</h3>
<ol>
<li><b>Add experiment</b> — Model module, or the empty viewport.</li>
<li><b>Import data</b> — Experiment → Add load steps (Vic-3D CSV + synchronized
machine tables, converted to measurement MAT) or Import MAT files
(natural-sorted files).</li>
<li><b>Mesh</b> — Mesh module: Nx / Ny, bounding box, or Fit mesh to DIC.</li>
<li><b>BCs</b> — select <i>Boundary conditions</i> in the tree, then add
prescribed, linking, or external conditions. One dialog can tick several
DOFs (W, W<sub>X</sub>, …) and apply them to a named edge <i>or</i> to
individual mesh nodes (tab junctions). Selected nodes highlight in the
3D viewport with their ids; click a mesh node to add it. Each DOF becomes
its own tree entry — that is the only place to edit DOF, region, load type,
or prescribed value. Copy BCs from another experiment on the same panel.
Ctrl-click a glyph to select it; Pick nodes adds mesh nodes to the selected BC.</li>
<li><b>Analysis window</b> — on <i>Load steps</i>, set start / end / reference
and tick Include in identification on the experiment. Under Applied loads,
assign each .mat <code>F</code> component to an EXTERNAL group (not necessarily
list order).</li>
<li><b>Check consistency</b> — Job page shows pre-flight errors and BC coverage;
VFM menu also runs the check before the first solve.</li>
<li><b>Identify</b> — F5. Watch the Job dock; Cancel stops the solver process.</li>
<li><b>Virtual fields</b> — Results → Virtual fields. Experiment, load step,
View, Quantity, and 3D unknown are in Properties. 3D uses linked VTK cameras
when the viewport is on; 2D contour stays matplotlib.</li>
<li><b>Post-processing</b> — plot View combos live in Properties. Stress
resultant space uses VTK for the 3D cloud (orbit / pan / zoom) and matplotlib
for 2D. Log and Jobs hide in Results unless a job is running. Save as
<code>.vfm</code> to keep θ, virtual fields, identifiability, UQ, and the last
VF view.</li>
</ol>
""".strip()


class WorkflowDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{APP_NAME} workflow")
        self.setMinimumSize(480, 360)
        label = QLabel(WORKFLOW_TEXT)
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.RichText)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        layout = QVBoxLayout(self)
        layout.addWidget(label, 1)
        layout.addWidget(buttons)
