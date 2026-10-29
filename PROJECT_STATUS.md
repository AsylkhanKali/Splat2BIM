# Splat2BIM: project status and deliverables

**Ready as a research prototype and portfolio demonstration.** Seven outdoor
captures are registered to their supplied LAS references, inspected against three
BIM revisions, and compared using geometric, semantic, and structural methods.
This is not a survey-certified construction-monitoring product.

The viewer, principal reports, and their presentation generators use English.
Earlier development reports remain in their original language as an archive.
The [README](README.md) is the portfolio entry point; [data setup](docs/DATA.md)
explains how to restore the excluded scene files.

## Delivered

| Deliverable | Entry point |
|---|---|
| Seven-date 3D viewer with LAS and BIM layers | [Local viewer](http://127.0.0.1:8765/viewer/) |
| Baseline registration and BIM comparison | [Seven-date report](results/multidate/REPORT.md) |
| BIM revision/geometry inspection | [0929 BIM report](results/0929_bim/BIM_CHECK.md) |
| Common-area diagnostics | [Shared-cell report](results/common_areas/REPORT.md) |
| Roof/riverside controls excluding work-site equipment | [Stable-region evaluation](results/static_structures/REPORT.md) |
| Cross-date LAS elevation analysis | [Elevation-correction test](results/las_vertical/REPORT.md) |
| Building specialist and correspondence-bank comparison | [Semantic fusion](results/semantic_fusion/REPORT.md) |
| Recognition using Gaussian-rendered views | [3DGS recognition](results/semantic_3dgs/REPORT.md) |
| Registration repeated with the new labels | [Seven-date label comparison](results/semantic_3dgs_registration/REPORT.md) |
| Persistent structures passing the unchanged strict gates | [Structural support](results/stable_structures/REPORT.md) |
| Optional local accuracy refinement | [Refinement report](results/reviewed_structural_refinement/REPORT.md) |
| Camera localization from a photograph | [Camera report](results/camera_localization/REPORT.md) |
| Detailed experimental history | [Research notes](RESEARCH_NOTES.md) |

Capture IDs are **0331, 0504, 0609, 0708, 0723, 0831, 0929**. The working full
Gaussian exports remain in `results/0929_baseline/` and `results/multidate/<date>/`.
Those multi-gigabyte files are local-only. The repository includes code, selected
figures, reports, tables, region definitions, and local/global transforms.

## What the results establish

### Registration and BIM

Every capture is aligned to its own LAS. The 0929 baseline uses RTK metadata and
COLMAP camera poses; the six other dates use geometry, with a documented visual
seed in the original 0708 baseline. Full Gaussian export preserves positions,
orientations, scales, opacity, and directional color consistently.

The local origin is **[825800, 842100, 0]**, in metres. BIM uses the same origin
without an additional date-specific fit. A global transformation already includes
that offset; do not add it twice. The authoritative BIM-revision/date mapping and
independent horizontal/vertical control remain unconfirmed.

### Recognition and strict structural support

The LoveDA building specialist initially improved recognition in LAS but missed
the 3DGS control roof. Gaussian rendering with covariance, opacity, and degree-3
spherical harmonics improved control-roof recognition to **59.8–99.7% across seven
dates**. This is a local diagnostic fraction, not whole-scene segmentation IoU.

Registration with the new labels passed primary checks on **7/7** dates and strict
checks on **5/7**. 0708 lacked sufficient transverse support spread; 0723 lacked
sufficient overlap on a third independent fragment. Those results are retained
as an earlier experiment, not presented as the current strict outcome.

Independent surface growth, measured infill, and alternative extents subsequently
recovered support on **7/7 dates without lowering thresholds**. Each physical
surface counts once and must persist in at least three LAS dates. The selected
0708/0723 supports persist in all seven. This strengthens support for the fit;
final geometric poses remain effectively unchanged from the previous method.

### Optional reviewed-region accuracy refinement

Using the existing manual roof/riverside region definitions, the mean of the two
control-region medians across seven dates decreases from **0.214 to 0.205 m
(3.9%)**. Strict acceptance remains **7/7**. Five dates improve by more than 1 mm;
0609 changes by less than 1 mm and 0929 retains its pose.

This is a local tradeoff. Whole-scene medians increase by **1.7–8.8 mm** on changed
dates, with P90 increases above 1 cm on 0331 and 0504. On the stronger spatial roof
holdout, four of five adequately sampled dates improve and 0609 worsens by 3.3 mm.
Spatial riverside samples are insufficient on every date. New matrices remain
separate; the working scene exports and BIM were not replaced.

Training and reported control records are disjoint, but the same region
definitions provide supervision. This is weaker than evaluation on new structures
or a new site. Two earlier automatic refinement variants are retained, including
the context-guarded variant that kept all seven parent poses.

### Image-to-model camera localization

The 0929 experiment uses **12 query images and 112 reference images**. Each query
and its ±10 neighboring frames are excluded from the reference index. Retrieval,
feature matching, LAS depths, and PnP-RANSAC estimate the camera using known lens
calibration, without the query COLMAP pose as an input.

All **12/12** candidates pass the predefined internal gate. Relative to COLMAP,
median camera-center difference is **0.167 m** (maximum **0.486 m**) and median
rotation difference is **0.055°** (maximum **0.212°**). Held-out matches within
3 pixels range from **68–94%**, with median **89%**, at 1200 × 900 resolution.
These are internal-consistency measurements: source photos may also have
contributed to LAS/COLMAP reconstruction.

## Verification and presentation provenance

The numerical suite contains **79 tests**, covering known-transform recovery,
Gaussian covariance/color preservation, spatial sampling, correspondence gates,
camera localization, rendering, and guarded refinement. Full data-dependent
verifiers previously recomputed the experiment fits, sample identities, strict
checks, and errors; the final refinement verification preserved 91 historical
file hashes.

The English presentation update changes prose, labels, figures, and report-source
hash metadata. It does not change numeric table values, fit matrices, region
masks, or threshold parameters. Original validation and delivery receipts describe
the original runs; the pre-translation research snapshot remains available at
commit `7bed2a70dff8cb857de15a8ced653b8f59b3e6ff`. See
[presentation verification](docs/PRESENTATION.md) for the distinction between
historical numerical verification and the current English publication checks.

## Run the demonstration

Restore or generate the large outputs as described in [DATA.md](docs/DATA.md),
then start the loopback-only server from the repository root:

```sh
.venv/bin/python scripts/serve_viewer.py
```

Open the [viewer](http://127.0.0.1:8765/viewer/), choose a capture date, and toggle
3DGS, LAS, or BIM. Under Alignment checks, select Stable structures or Shared
20 × 20 m cells. Report links open the semantic, structural, accuracy, and camera
inspection pages. Only one full Gaussian scene is loaded at a time.

## What remains for stronger claims

- Independent surveyed control, confirmed vertical datum, and authoritative BIM
  revisions are needed for absolute accuracy or construction-tolerance claims.
- Reviewed semantic ground truth and evaluation on a separate site are needed to
  establish segmentation quality and generalization.
- The local refinement has a whole-scene tradeoff and manual supervision; it is
  not established as a globally better automatic registration method.
- Cross-date LAS differences cannot be corrected confidently with one roof-derived
  height offset. Agreement with same-date LAS and agreement between dates are
  separate questions.

The current scope is complete for demonstrating the implemented methods, measured
results, and limitations. Production readiness and independent survey accuracy
have not been established.
