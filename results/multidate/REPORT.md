# All seven captures: 3DGS → LAS → BIM

All dates share one metre coordinate frame: LAS/BIM world = local + `[825800,842100,0]`. Original files are preserved. BIM is not fitted to individual captures; the unchanged v3 is used for comparison. Official BIM-revision assignments to dates remain unconfirmed.

## 3DGS agreement with same-date LAS

| Date | Gaussians | Median, m | P90, m | Within 1 m | XY coverage | Median change vs 0929, m |
|---|---:|---:|---:|---:|---:|---:|
| 0331 | 3,000,000 | 0.266 | 1.151 | 88.3% | 98.7% | +0.047 |
| 0504 | 2,929,187 | 0.196 | 0.758 | 93.1% | 99.3% | -0.023 |
| 0609 | 2,888,037 | 0.248 | 0.807 | 92.8% | 99.7% | +0.029 |
| 0708 | 2,380,897 | 0.249 | 1.018 | 89.8% | 99.4% | +0.030 |
| 0723 | 2,580,522 | 0.209 | 0.759 | 93.5% | 100.0% | -0.010 |
| 0831 | 2,882,481 | 0.230 | 0.772 | 94.4% | 100.0% | +0.011 |
| 0929 | 2,874,168 | 0.219 | 0.745 | 93.3% | 99.1% | +0.000 |

Every tenth Gaussian is sampled, with opacity logit > 0 and the outer 0.5% removed on each source axis. About 25% of spatial tiles, each 0.4 source units wide, is excluded from fitting. Median/P90 and fraction within 1 m use held-out centers inside LAS XY coverage (nearest XY point < 1 m). Source-unit size and tile shape differ between dates. This diagnoses agreement with each date-specific LAS; it is neither independent survey accuracy nor a measurement of construction change.
0929 retains the RTK-camera/COLMAP initialization. The other six dates lack camera exports: four geometric orientations are tested, then scale, rotation, and translation are refined. Coverage differences made this insufficient for 0708, so a rough top-view seed from recognizable objects was followed by the same training-tile fit. align_dates.py contains the seed; 0708/landmark_candidate.json and 0708/geometric_rejected.json retain landmarks and rejected hypotheses. These landmarks are not survey control. Color and visual checks help distinguish false local minima.

Export transforms coordinates, quaternions, log scales, and color spherical harmonics. All original Gaussians are retained, including those excluded from fitting. Local/global matrices, input SHA-256 hashes, and reopened-export checks are saved for every date.

A diagnostic guard prevents export of clearly poor candidates: training median > 0.5 m or median RGB difference > 0.35 produces rejected status. This is an internal gross-error filter for this dataset, not an agreed accuracy tolerance.

## One BIM v3 compared with LAS from every date

Median of element-wise median surface distances to LAS, in metres. Each element uses the same 500 surface samples across dates, and each element has equal weight.

| Group | 0331 | 0504 | 0609 | 0708 | 0723 | 0831 | 0929 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Decks | 5.03 | 4.40 | 4.60 | 0.82 | 0.51 | 0.73 | 0.72 |
| Canopies | 8.41 | 6.83 | 7.37 | 3.85 | 2.44 | 0.54 | 0.37 |
| Piers | 1.92 | 1.53 | 0.93 | 0.92 | 0.94 | 0.76 | 0.71 |
| Pile caps | 2.22 | 0.99 | 2.26 | 3.23 | 3.07 | 3.44 | 3.79 |
| Kiosks | 1.34 | 1.40 | 1.59 | 1.83 | 2.32 | 0.93 | 0.90 |
| Underground piles | 20.50 | 18.44 | 19.99 | 20.13 | 18.95 | 17.48 | 18.85 |

Samples include underside, hidden, and underground surfaces. Captures also differ in coverage and construction stage. Large distances do not prove that a structure is missing or registration is wrong. This table cannot yield a completion percentage; changes must be measured separately on commonly observed regions. BIM statistics use the original LAS and are separate from 3DGS registration errors.

## Viewing and reproduction

http://127.0.0.1:8765/viewer/ changes 3DGS and its LAS together while preserving the viewpoint and BIM revision. Full clouds load one at a time; the viewer subsamples LAS by 10, while numerical checks use the full LAS.

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python scripts/align_dates.py
.venv/bin/python scripts/export_dates.py
.venv/bin/python scripts/compare_dates.py
.venv/bin/python scripts/preview_dates.py
.venv/bin/python scripts/serve_viewer.py
```

## Completed-result checks

26 October 2026: 7 numerical tests passed; all 6 exported PLYs and 14 local resource URLs were checked. All 7 dates loaded and were inspected in Spark. View preservation, paired LAS switching, and selected BIM-element metrics were checked; the final console had no errors or warnings. Maximum inspected export-coordinate rounding was 0.000030 m; this is file-writing precision, not registration accuracy. See validation.json.
