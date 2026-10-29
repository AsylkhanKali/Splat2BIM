# Semantics: structure recognition and the 0723 rejection

Current combined: 7/7; Previous semantics: 6/7; Geometry + pair bank: 5/7; Previous labels + pair bank: 7/7; New labels + pair bank: 7/7.

Accepted median improvements above 1 cm relative to the current method: **0**, regressions: **0**. This threshold describes difference magnitude, not statistical significance. The working combined method remains the default.

## All seven dates compared

| Date | Current median / P90, m | Previous semantics: acceptance | Previous labels + pair bank: acceptance | New semantic median / P90, m | New acceptance |
|---|---:|---|---|---:|---|
| 0331 | 0.272 / 1.148 | yes | yes | 0.272 / 1.148 | yes |
| 0504 | 0.205 / 0.754 | yes | yes | 0.205 / 0.754 | yes |
| 0609 | 0.253 / 0.805 | yes | yes | 0.253 / 0.805 | yes |
| 0708 | 0.255 / 1.006 | yes | yes | 0.255 / 1.006 | yes |
| 0723 | 0.213 / 0.753 | rejected | yes | 0.213 / 0.753 | yes |
| 0831 | 0.234 / 0.752 | yes | yes | 0.234 / 0.752 | yes |
| 0929 | 0.219 / 0.742 | yes | yes | 0.219 / 0.742 | yes |

Acceptance uses the previous internal scale, geometry, RGB, and minimum-three-spaced-pairs gates. Rejected matrices, where available, remain in the CSV with explicit status. Distances are from Gaussian centers to the supplied LAS on identical spatially held-out points with the previous fixed coverage; this is not independent survey accuracy.

The current method here is the automatic selection from `object_recovery/selection.json`; it differs from the historical full viewer exports. Neither the current final matrix, RTK, nor the manual 0708 seed is used to select new starts.

## What changed

1. A building specialist is added: [SegFormer-B2 trained on LoveDA](https://huggingface.co/wu-pr-gw/segformer-b2-finetuned-with-LoveDA), revision `5c74556c08bebb5f45f50b6f78f61a62c5d220c7`. [LoveDA](https://github.com/Junjue-Wang/LoveDA) includes a building class in overhead imagery. The upstream configuration has eight outputs, including Ignore/Background, and its normalization is used. No repository code is executed; PyTorch loads weights through `weights_only=True`, and their SHA-256 is verified.

2. Views use the previous training points. Image gaps are filled from the nearest observed color and a 3×3 median filter. The source radius follows the 0.4-source-unit holdout tile size; the LAS radius is 24 pixels. This approximates context rather than reconstructing a surface: filled pixels never become 3D points. Artificial texture extensions are visible near boundaries.

3. LoveDA supplies building predictions at probability ≥0.6. The previous drone model supplies other classes. Confident vegetation, water, and moving-object labels are protected from replacement. Previous structure labels become either the new prediction or uncertainty. Replacing all classes with LoveDA was rejected before the final seven-date comparison because some cloud pairs lacked enough shared classes. Raw predictions for every date remain in `semantic_structures`, together with the diagnostic 0331 registration run.

4. Before fitting each selected start, pairs proposed by initial hypotheses within RMS <2 m on the same training anchors are pooled. Each pair records its proposing hypothesis IDs. After fitting, only these identities can pass the existing one-to-one geometric assignment, bidirectional overlap, and spacing gates. No new pairs are added from the final pose.

All three new variants use the same pool, four-start budget, ICP stages, and final refinement. The old-labels-plus-pair-bank control separates the 0723 fix from the new network. Held-out errors do not set parameters. This is another development experiment on a known site, not a new independent test.

## Why 0723 was rejected

The previous geometric and semantic results differed by at most 8.53×10⁻¹⁴ in their matrix entries: their poses were effectively identical. However, start `recovery_366` retained five pairs while neighboring `recovery_373` retained six. Two previously confirmed pairs are too close to count as separate spatial supports.

The bank around starts 558 and 564 adds one previously proposed pair, `p8_c16 → p8_c39`. Diagnostics of the old final pose show that it supplies a third spaced support. Its fractions within 1 m are 71.2% forward and 55.4% reverse, with center distance 4.895 m against a 5 m limit. This is a narrow margin, not evidence of high reliability. `0723_failure_analysis.json` preserves the details.

Strict check requiring at least three pairs with forward overlap ≥80%: Current combined 4/7, Previous semantics 5/7, Geometry + pair bank 3/7, Previous labels + pair bank 5/7, New labels + pair bank 5/7. The new variant still fails strict checks on 0708 and 0723. More supporting pairs at the same pose do not imply higher accuracy.

## Recognition evaluation

| Date | LAS roof: old → new prediction | 3DGS roof: new prediction | False structures in LAS vegetation |
|---|---:|---:|---:|
| 0331 | 0.0% → 64.4% | 0.0% | 0.0% |
| 0504 | 0.0% → 90.8% | 0.0% | 0.0% |
| 0609 | 0.0% → 89.3% | 0.0% | 0.0% |
| 0708 | 0.0% → 0.0% | 0.0% | 0.0% |
| 0723 | 0.0% → 0.0% | 0.0% | 0.0% |
| 0831 | 0.0% → 99.3% | 0.0% | 0.0% |
| 0929 | 0.0% → 8.0% | 0.0% | 0.0% |

![Control-roof recognition](recognition.png)

These are visible-point fractions inside fixed diagnostic volumes, not IoU or global classification accuracy. The regions are the previously inspected roof and an RGB-reviewed vegetation strip without buildings. `recognition.csv` records counts and old/new labels for both clouds. The previous automatic matrix only locates diagnostic volumes in the source; segmentation and registration do not read these volumes.

**Recognition improves partially:** the LAS roof is detected on five dates, but the 3DGS roof is still missed on all seven. There are too few confident building labels for a shared class on dates 0609, 0708, 0723, 0831. Noisy Gaussian centers and missing tiles differ strongly from the training imagery. Reviewed labels and adaptation to these projections, or better 3DGS views, are needed; more orange points do not prove recognition accuracy.

[RGB review of the negative region](negative_region_review.png). One roof and one vegetation strip cannot establish precision and recall for the whole site. The models have no separate crane class. Local stable-structure evaluation uses the previous equipment-exclusion masks.

## Stable structures and reproduction

`static_structures.csv` records the roof and riverside structure using both unused records and strict spatial samples. The first sample has nearby training points; the second is often small on narrow structures. Empty metrics are not replaced with zeros. BIM and the working full PLYs are not moved.

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python scripts/prepare_semantic_structures.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python scripts/prepare_semantic_fusion.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python scripts/align_semantic_fusion.py
.venv/bin/python scripts/check_semantic_recognition.py
.venv/bin/python scripts/report_semantic_fusion.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -m unittest discover -s scripts -p 'test_*.py' -v > results/semantic_fusion/tests.log 2>&1
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python scripts/verify_semantic_fusion.py
```

The saved geometric pools and previous drone labels are required. No packages were added. Weights, parameters, and code hashes were fixed before the final comparison. `validation.json` records numerical re-verification; `http_checks.json` records local-page checks. The [previous experiment](../semantic_initial/REPORT.md) is retained.

In `*_labels.npz`, `labels`, `confidence`, and `label_origin` describe the final decision. `probabilities` retains the original drone-model probabilities, while `building_expert_probabilities` holds LoveDA predictions. These are separate distributions, not a single calibrated probability for the fused method. `label_origin`: 0 = previous class, 1 = building specialist, 2 = previous building label replaced with uncertainty.
