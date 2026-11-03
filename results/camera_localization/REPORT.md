# Camera pose from a single photograph

**0929: 12 of 12 images passed the internal quality gate.** The query is matched to a reference image set, then its camera pose is estimated from LAS correspondences. BIM remains in the original shared coordinate frame.

## Protocol

Twelve available images are sampled uniformly by filename. These queries and ±10 neighboring frames are excluded from the reference set; protocol.json records the exact split. Reference cameras are calibrated through the previous COLMAP → LAS transform. Visible LAS depths provide 2D–3D correspondences. The method uses SIFT/RootSIFT, visual retrieval of five reference images, mutual matches with distance ratio <0.75, PnP-RANSAC, and LM refinement. Query-camera coordinates and RTK are not used for retrieval or pose fitting. Known query-lens calibration is used.

Approximately one quarter of the 150 × 150 pixel image blocks is excluded from PnP and LM. Acceptance requires ≥25 training inliers, an inlier fraction ≥20%, ≥10 control matches, ≥35% of control matches within 3 pixels, an inlier convex hull covering ≥5% of the image, and positive depth. The 3-pixel threshold applies at 1200 × 900, not the original 4032 × 3024 resolution. The full control-error median includes false matches.

All 12 results are reported, including any rejections. Parameters and gates were fixed before the first run of this series. This is a research experiment on a known site. Withheld photos may have contributed to the original LAS/COLMAP reconstruction; distance to a COLMAP camera measures internal consistency, not independent absolute accuracy. Blocks within one image are not statistically independent either.

## Results

| Image | Decision | Matches / inliers | Control ≤3 px | Center difference vs COLMAP, m | Rotation difference, ° |
|---|---|---:|---:|---:|---:|
| 1 | candidate | 607 / 384 | 88% | 0.114 | 0.031 |
| 2 | candidate | 728 / 511 | 74% | 0.117 | 0.036 |
| 3 | candidate | 922 / 588 | 89% | 0.152 | 0.045 |
| 4 | candidate | 794 / 531 | 90% | 0.190 | 0.052 |
| 5 | candidate | 1244 / 847 | 94% | 0.119 | 0.037 |
| 6 | candidate | 652 / 442 | 82% | 0.258 | 0.104 |
| 7 | candidate | 740 / 399 | 91% | 0.336 | 0.155 |
| 8 | candidate | 202 / 84 | 68% | 0.486 | 0.212 |
| 9 | candidate | 1045 / 768 | 91% | 0.202 | 0.059 |
| 10 | candidate | 527 / 303 | 93% | 0.135 | 0.072 |
| 11 | candidate | 1755 / 1210 | 94% | 0.058 | 0.050 |
| 12 | candidate | 411 / 249 | 85% | 0.181 | 0.103 |

## Photo overlays

Cyan points show subsampled LAS without full surface-occlusion handling. Yellow lines show BIM v3 edges from the estimated camera, including hidden structures. Lens distortion is applied to projections; the original photo is not rectified. BIM v3 is illustrative; its validity for 0929 is unconfirmed. BIM agreement is not part of the camera gate. The wireframe is not a mask of completed structures; mismatch may reflect construction stage, design revision, or alignment error.

### Image 1: DJI_20260929091641_0042_V.jpeg

![Photo, LAS and BIM](figures/query_00.png)

### Image 2: DJI_20260929091902_0178_V.jpeg

![Photo, LAS and BIM](figures/query_01.png)

### Image 3: DJI_20260929092113_0315_V.jpeg

![Photo, LAS and BIM](figures/query_02.png)

### Image 4: DJI_20260929092317_0452_V.jpeg

![Photo, LAS and BIM](figures/query_03.png)

### Image 5: DJI_20260929092506_0589_V.jpeg

![Photo, LAS and BIM](figures/query_04.png)

### Image 6: DJI_20260929092710_0726_V.jpeg

![Photo, LAS and BIM](figures/query_05.png)

### Image 7: DJI_20260929092906_0863_V.jpeg

![Photo, LAS and BIM](figures/query_06.png)

### Image 8: DJI_20260929093111_1000_V.jpeg

![Photo, LAS and BIM](figures/query_07.png)

### Image 9: DJI_20260929093325_1137_V.jpeg

![Photo, LAS and BIM](figures/query_08.png)

### Image 10: DJI_20260929093550_1274_V.jpeg

![Photo, LAS and BIM](figures/query_09.png)

### Image 11: DJI_20260929093957_1411_V.jpeg

![Photo, LAS and BIM](figures/query_10.png)

### Image 12: DJI_20260929101456_0136_V.jpeg

![Photo, LAS and BIM](figures/query_11.png)

## Files and reproduction

`poses.json` contains world→camera matrices (X right, Y down, Z forward), centers in local metres, K, and distortion coefficients. The local origin is [825800, 842100, 0]. `query_*_matches.npz` stores control/training pairs, inlier indices, and errors. `inputs.json` and the cache contain input and parameter SHA-256 hashes. The status candidate_for_review means internal acceptance for inspection, not survey approval.

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python scripts/run_camera_localization.py
.venv/bin/python scripts/report_camera_localization.py
```

The command below processes one image. The example calibration applies only to the first control photo; a different lens/image needs its own calibration. The output file must be new. Images already in the reference index are rejected to avoid presenting self-retrieval as localization.

```sh
.venv/bin/python scripts/localize_photo.py --image assets/0929/photos/DJI_20260929091641_0042_V.jpeg --calibration results/camera_localization/example_calibration.json --output results/camera_localization/my_pose.json
```

The method assumes known lens calibration, overlap with reference photos, and already aligned reference cameras. Equivalent data is unavailable for the other six dates. Transfer to a new site, an unknown lens, or a substantially different viewpoint is untested.

Primary references: [COLMAP text format](https://colmap.github.io/format.html), [OpenCV PnP and LM](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html), [SIFT](https://docs.opencv.org/4.x/d7/d60/classcv_1_1SIFT.html).
