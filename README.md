<div align="center">

# Splat2BIM · 0929

**Measure what PCA, ICP and semantic labels actually contribute.**

A single-date research study of 3D Gaussian Splatting registration to LAS,
BIM inspection, and calibrated photo-to-camera-pose estimation.

[Controlled report](results/ablation/inspection.html) · [Results](results/ablation/REPORT.md) · [Pipeline](results/ablation/pipeline.svg) · [Camera experiment](results/camera_localization/REPORT.md) · [Setup](docs/REPRODUCING.md)

</div>

![Controlled comparison](results/ablation/comparison.png)

## Controlled experiment

The published study contains **only capture 0929**. It compares:

1. **PCA only** — four proper sign hypotheses, selected by training geometry.
2. **PCA + ICP** — the same PCA pool, geometric selection and robust similarity refinement.
3. **PCA + semantic ranking + ICP** — the same pool and refinement budget, with class-balanced SegFormer label distances added to candidate ranking.

All variants use the same **48,081 held-out Gaussian centers** and LAS reference.
The primary comparison refines one candidate. A **14-setting × 3-seed ablation**
tests candidate budget, semantic weight, confidence filtering, trimming, voxel
resolution and shuffled source labels. No RTK, fitted baseline pose or manual
initialization is used in these controlled fits.

| Primary method | Mean held-out median, m | Mean P90, m |
|---|---:|---:|
| PCA only | 2.278 | 14.703 |
| PCA + ICP | 1.350 | 8.140 |
| PCA + semantic ranking + ICP | 1.350 | 8.140 |

Values average the per-run metrics over three sampling seeds. **ICP improves
reference agreement; semantic ranking provides no additional final-pose benefit
in this experiment.** Its selected final matrices match the geometric branch.
Higher fitting retention improves the observed metrics in the settings sweep;
this is exploratory evidence, not a newly validated default.

The report includes every run, stage-by-stage metrics, paired spatial-block
bootstrap intervals, class-label images, candidate traces, and **local/global
4×4 transformation matrices**. The [numerical verifier](results/ablation/verification.json)
recomputes distances and candidate selection from the saved inputs and matrices.

## What is separate

- **Viewer baseline:** the historical 0929 RTK/COLMAP-assisted full-scene export.
  It is preserved for inspection and has not been replaced by the PCA experiment.
  Its original overlap-filtered median of about 0.219 m is not directly comparable
  to the new full-cohort metrics above.
- **BIM:** three IFC revisions are placed in the shared local frame. BIM is not
  used as a target or an accuracy certificate for the controlled registration.
- **Photo-to-camera pose:** SIFT/RootSIFT retrieval and matching, LAS depths,
  EPnP-RANSAC and LM refinement. All 12 historical queries passed the internal
  gate; median camera-center difference from COLMAP was 0.167 m. Reference camera
  poses are inputs; query pose is used only for evaluation.

## Run locally

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
npm ci --prefix viewer
python -m unittest discover -s scripts -p 'test_*.py' -v
python scripts/serve_viewer.py
```

Open [the local viewer](http://127.0.0.1:8765/viewer/) or
[the controlled HTML report](http://127.0.0.1:8765/results/ablation/inspection.html).
The report and its figures work without the raw dataset. The viewer requires the
large local scene exports. GitHub displays Markdown; serve the repository locally
to render the HTML report. See [reproduction instructions](docs/REPRODUCING.md).

## Limits

This is one previously explored capture, not an independent new-site test.
Pretrained SegFormer labels and rendering settings were developed previously.
Held-out centers do not enter fitting, but spatial dependence, overlapping
Gaussian footprints, and shared reconstruction photographs remain. LAS is a
reference, not surveyed ground truth. Reported distances are reference agreement,
not certified positional accuracy, segmentation IoU or construction tolerances.
No further dataset is assumed to be available.

## Publication scope

Only 0929 results and supporting code are published. Raw PLY/LAS/IFC inputs,
photographs, semantic arrays, pretrained weights and full scene exports stay
local. The public history was replaced with this single-date edition; the full
research archive is retained privately on the owner's computer.
