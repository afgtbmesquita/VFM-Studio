# VFM projects

- `campaign.vfmproj.json` is the verified two-coupon raw-frame project.
- `campaign_clamped_bottom.vfmproj.json` and `campaign_clamped_top.vfmproj.json` are the
  one-loaded-cut boundary-condition sensitivity projects.
- `samples_ab_all.vfmproj.json` is the Samples A/B stacked campaign (six pure-bending
  coupons plus A1/B1 edge twist) with a complete bottom-edge encastre and load-cell
  EXTERNAL virtual work on the top cut only. Rebuild with
  `python Samples_A_B/VFM_analysis/make_gui_project.py`.

Rebuild and numerically verify the canonical Pre-Conference project with
`python -m workflows.make_gui_project`. Rebuild the clamped projects with
`python -m workflows.compare_bc_forms --write-projects`.

