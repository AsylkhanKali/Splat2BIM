# Published result evidence

This directory contains a compact selection from the completed local experiments:

- Markdown reports and CSV comparisons.
- Coordinate transforms, region definitions, and experiment protocols.
- Summaries, input manifests, and recorded validation receipts.
- All figures referenced by the eleven principal reports and six HTML pages.
- English viewer screenshots used by the project overview and region report.

The full local results directory is approximately 6.7 GB. Gaussian/LAS/IFC-derived
geometry, NumPy arrays, detailed JSON fit traces, and unselected development images
are excluded from Git. Earlier reports may refer to excluded files; restore the
original inputs and reproduce the corresponding experiment to use those links.
The principal reports and their published figures can be read without the data.

## Start here

| Experiment | Evidence |
|---|---|
| Seven-date registration and BIM comparison | [Report](multidate/REPORT.md) |
| Fixed roof/riverside control regions | [Report](static_structures/REPORT.md), [definitions](static_structures/regions.json) |
| BIM revisions and shared coordinate frame | [Report](0929_bim/BIM_CHECK.md) |
| Common-area comparison | [Report](common_areas/REPORT.md) |
| Cross-date reference elevation | [Report](las_vertical/REPORT.md) |
| Building specialist and correspondence bank | [Report](semantic_fusion/REPORT.md), [HTML](semantic_fusion/inspection.html) |
| Recognition from Gaussian-rendered views | [Report](semantic_3dgs/REPORT.md), [HTML](semantic_3dgs/inspection.html) |
| Gaussian-view semantic registration | [Report](semantic_3dgs_registration/REPORT.md) |
| Persistent structures, strict acceptance on 7/7 dates | [Report](stable_structures/REPORT.md), [CSV](stable_structures/summary.csv) |
| Optional reviewed-region accuracy refinement | [Report](reviewed_structural_refinement/REPORT.md), [CSV](reviewed_structural_refinement/summary.csv), [validation](reviewed_structural_refinement/validation.json) |
| Image-to-model camera localization | [Report](camera_localization/REPORT.md), [poses](camera_localization/poses.json) |

Principal reports are in English. Earlier development reports preserve their original Russian prose.
The [main README](../README.md) provides the current English overview and explains
the limitations of each headline result.

`validation.json` files record checks performed with the full local dataset. They
are evidence of those runs, not a claim that a data-free clone can run the complete
verification pipeline. The synthetic unit tests can run without the dataset.

See the [data and reproduction guide](../docs/DATA.md) before running experiments.
The [English presentation record](../docs/PRESENTATION.md) explains current
publication checks and the preserved historical receipts. GitHub displays
Markdown directly; serve the repository locally to browse the HTML pages.
