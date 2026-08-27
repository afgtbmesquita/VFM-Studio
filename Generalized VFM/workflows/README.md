# Workflows

Run these commands from the `Generalized VFM` directory.

| purpose | command |
|---|---|
| known-answer GLS validation | `python -m workflows.validate_gls` |
| constitutive-frame validation | `python -m workflows.validate_rotation` |
| two-coupon campaign | `python -m workflows.run_campaign_gls --all` |
| three-coupon campaign | `python -m workflows.run_campaign_45 --all` |
| two-coupon BC study | `python -m workflows.bc_study` |
| 45° BC/trim study | `python -m workflows.bc_study_45` |
| assumption audit | `python -m workflows.audit_assumptions` |
| audit figures | `python -m workflows.make_audit_figures` |
| campaign figures | `python -m workflows.make_figures` |
| rebuild/verify GUI project | `python -m workflows.make_gui_project` |
| compare and write clamped projects | `python -m workflows.compare_bc_forms --write-projects` |

`campaign_common.py` is the single source of campaign paths, coupon definitions, material
orientation, mesh setup, and standard output directories. Workflows write tables and figures
under `../results/` and project definitions under `../projects/`.

