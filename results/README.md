# Published result evidence

This directory contains a compact selection from the completed local experiments:

- Markdown reports and CSV comparisons.
- Coordinate transforms, region definitions, and experiment protocols.
- Summaries, input manifests, and recorded validation receipts.
- Selected plots and viewer screenshots used by the project overview.

The full local results directory is approximately 6.7 GB. Gaussian/LAS/IFC-derived
geometry, NumPy arrays, detailed JSON fit traces, rendered image galleries, HTML
inspection pages, and most development images are excluded from Git. A report may
refer to one of these generated files; restore the original inputs and reproduce
the corresponding experiment to use that link locally.

## Start here

| Experiment | Evidence |
|---|---|
| Seven-date registration and BIM comparison | [Report](multidate/REPORT.md) |
| Fixed roof/riverside control regions | [Report](static_structures/REPORT.md), [definitions](static_structures/regions.json) |
| Gaussian-view semantic registration | [Report](semantic_3dgs_registration/REPORT.md) |
| Persistent structures, strict acceptance on 7/7 dates | [Report](stable_structures/REPORT.md), [CSV](stable_structures/summary.csv) |
| Optional reviewed-region accuracy refinement | [Report](reviewed_structural_refinement/REPORT.md), [CSV](reviewed_structural_refinement/summary.csv), [validation](reviewed_structural_refinement/validation.json) |
| Image-to-model camera localization | [Report](camera_localization/REPORT.md), [poses](camera_localization/poses.json) |

Detailed reports preserve the original research record, including Russian prose.
The [main README](../README.md) provides the current English overview and explains
the limitations of each headline result.

`validation.json` files record checks performed with the full local dataset. They
are evidence of those runs, not a claim that a data-free clone can run the complete
verification pipeline. The synthetic unit tests can run without the dataset.

See the [data and reproduction guide](../docs/DATA.md) before running experiments.
