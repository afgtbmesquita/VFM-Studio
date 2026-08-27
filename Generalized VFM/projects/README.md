# VFM projects

- `campaign.vfmproj.json` is the verified two-coupon raw-frame project.
- `campaign_clamped_bottom.vfmproj.json` and `campaign_clamped_top.vfmproj.json` are the
  one-loaded-cut boundary-condition sensitivity projects.

Rebuild and numerically verify the canonical project with
`python -m workflows.make_gui_project`. Rebuild the clamped projects with
`python -m workflows.compare_bc_forms --write-projects`.

