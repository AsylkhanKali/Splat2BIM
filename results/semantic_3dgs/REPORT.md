# 3DGS: structure recognition and strict failures

This experiment improves structure recognition in 3DGS. Strict statuses refer to the previous semantic registration: its transforms and geometric supports are preserved. Improved recognition alone does not prove better alignment.

The control roof received structure labels on 7/7 dates instead of 0/7. This is a fraction of selected visible points in one volume, not IoU or whole-model accuracy.

| Date | Roof points | Roof: before → after | False buildings in vegetation | Pairs with sufficient overlap | Strict check of previous pose |
|---|---:|---:|---:|---:|---|
| 0331 | 666 | 0.0% → 86.2% | 0.0% | 4 | passed |
| 0504 | 1269 | 0.0% → 99.7% | 0.0% | 10 | passed |
| 0609 | 1143 | 0.0% → 99.0% | 0.0% | 9 | passed |
| 0708 | 888 | 0.0% → 59.8% | 0.0% | 3 | failed |
| 0723 | 1319 | 0.0% → 91.9% | 0.0% | 2 | failed |
| 0831 | 317 | 0.0% → 85.5% | 0.0% | 5 | passed |
| 0929 | 284 | 0.0% → 80.6% | 0.0% | 6 | passed |

![Comparison](recognition.png)

## What changed

Sparse centers with color-filled gaps are replaced by orthographic Gaussian rendering: covariance is projected from scale and quaternion rotation, opacity is composited back to front, and color includes spherical harmonics up to degree 3. Resolution increases from 1024 to 2048 while preserving image boundaries. This is an orthographic view, not an exact rendering from the original photo camera.

LoveDA B2, normalization, 512-pixel tiles with four rotations, the 0.6 building threshold, and protection of confident drone vegetation/water/moving-object classes remain unchanged. New labels are assigned to exactly the previous visible 3D points; filled pixels create no geometry. LAS labels come from the preceding experiment.

Rendering uses only even original PLY records within the previous training bounds and outside spatial holdout tiles of 0.4 source units. Transparent Gaussians with alpha >1/255 are retained for context. Their footprints may cross excluded-tile boundaries, but control-center records are not read by the renderer. Final poses, BIM, and diagnostic volumes do not define the projection or prediction.

**Evaluation boundary:** resolution was selected after exploratory roof trials on 0723 and 0929; this is development on a known site. The same configuration was then applied to all seven dates. Roof results are not an independent test. Rendering method and resolution changed together, so their separate contributions are not measured. The previous automatic matrix is used only to locate diagnostic volumes.

[Negative-region RGB across seven dates](negative_region_review.png). The vegetation strip checks one type of false positive. It does not evaluate cranes, equipment, ground, or the whole site; the model has no separate crane class.

## Why the strict check fails

0708: three pairs meet the overlap gates, but their centers are almost collinear. The second singular value of centered source coordinates is 1.962274 m against a 2 m minimum; LAS gives 4.353667 m. Minimum spacing is about 59 m, so the issue is transverse spread. The 3.77 cm gap is in this support metric, not object-position error. The threshold is unchanged.

0723: only two pairs meet ≥80% forward and ≥50% reverse overlap within 1 m. The third spaced pair p8_c16 → p8_c39 gives 71.15% / 55.38%: 37 of 52 source points are within 1 m of the assigned fragment, while at least 42 are needed. Five confirmed points are missing; this does not justify adding artificial points.

All triples of eligible pairs were checked on both dates; none passes. An additional all-fragment match at the final pose also found none. That check is diagnostic only and does not decide acceptance.

All points in the third 0723 pair are within 1 m of the full LAS, but not all are near the assigned fragment. This indicates differences in fragment extent or identity: closeness to any LAS neighbor does not establish object correspondence. On 0708, pair p9_c36 → p6_c2 has only 44.52% reverse overlap; 87.2% of reverse misses fall in source holdout tiles. Removing these misses to gain acceptance is not valid.

The next justified step is to confirm a persistent support away from the near-straight line on 0708 and extract complete corresponding parts of the same object on 0723. Structure identities must be testable and fixed before fitting; pairs were not substituted after seeing results and thresholds were not relaxed.

![0708](../strict_diagnostics/0708_diagnostic.png)
![0723](../strict_diagnostics/0723_diagnostic.png)

## Files and reproduction

`summary.csv` compares seven dates; `recognition.csv` records control-region counts; `<date>/source_labels.npz` stores new source-point labels and separate LoveDA probabilities; `render_context.npz` stores original context indices; `segmentation_protocol.json` records parameters and hashes; `../strict_diagnostics/` stores pairs, triples, and thresholds. probabilities remains the previous drone-model distribution, building_expert_probabilities stores the separate LoveDA prediction, and labels/confidence/label_origin describe fusion.

```sh
.venv/bin/python scripts/prepare_semantic_3dgs.py
.venv/bin/python scripts/diagnose_strict_support.py
.venv/bin/python scripts/report_semantic_3dgs.py
.venv/bin/python -m unittest discover -s scripts -p 'test_*.py' -v > results/semantic_3dgs/tests.log 2>&1
.venv/bin/python scripts/verify_semantic_3dgs.py
```

The existing packages and weights are used. A clang++ compiler is required for the local renderer. Previous results and raw assets are preserved. validation.json records the checks. See the [previous registration comparison](../semantic_fusion/REPORT.md).
