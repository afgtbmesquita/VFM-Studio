# Results

Generated campaign artifacts are separated from solver and workflow source code.

| folder | contents |
|---|---|
| `two_coupon/` | two-coupon coefficient tables, step estimates, and captured console output |
| `three_coupon/` | three-coupon coefficient tables, step estimates, and captured console output |
| `audit/` | assumption-audit data and captured console output |
| `bc_studies/two_coupon/` | two-coupon BC-study CSV |
| `bc_studies/three_coupon/` | 45° BC-study CSV and captured console output |
| `figures/` | publication PDFs, PNGs, and figure-data tables |

The `*_output.txt` files are captured run logs. Workflow scripts write numerical data and
figures directly; capture stdout explicitly when a refreshed full log is required.

