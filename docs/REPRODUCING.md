# Reproducing the 0929 study

Use Python 3.11, the pinned requirements, and Node/npm for the viewer. Gaussian
rendering requires a C++17 compiler (`clang++` on the recorded macOS environment).
The public tests run without original assets or model weights.

## Input layout

```
assets/0929/0929.ply
assets/0929/0929.las
assets/0929/photos/
assets/0929/colmap/{cameras,images,points3D}.txt
assets/BIM/The Loop Federated Model v1.ifc
assets/BIM/The Loop Federated Model v2.ifc
assets/BIM/The Loop Federated Model v3.ifc
```

The PLY must contain the original Gaussian attributes, including rotations,
log-scales, opacity and spherical harmonics. No raw dataset download is provided.
All coordinates use metres relative to `[825800, 842100, 0]` after alignment.
A global transform already includes this origin; do not add it twice.

## Controlled registration ablation

Restore the two exact frozen semantic inputs privately:

```
results/segmentation/source_labels.npz
results/segmentation/target_labels.npz
```

Their hashes and those of the PLY/LAS are in
[the frozen protocol](../results/ablation/protocol.json). The inputs contain visible
training-point coordinates, fused class labels and confidence. They do not contain
fitted alignment matrices. The experiment reuses these labels rather than silently
changing segmentation while varying registration settings.

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  python scripts/run_registration_ablation.py --out results/ablation_new
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  python scripts/verify_registration_ablation.py --out results/ablation_new
python scripts/report_registration_ablation.py --out results/ablation_new
```

The runner freezes its protocol before fitting and refuses to reuse an existing
protocol directory. Evaluation always uses the same fixed source indices. The
verifier recomputes PCA hypotheses, training candidate scores, all checkpoint
metrics, input hashes, inverse transforms and local/global conventions. It also
checks stop reasons and iteration counts, independently probes two additional
terminal updates for stationarity and two-step oscillation, and compares the
historical transform on the same held-out cohort. It does not repeat the full
optimization. Run it before the report
generator, which reads these additional checks.

The primary limit is 500 iterations per gate; 30- and 200-step controls isolate
the budget effect. The previous stopping tolerances and distance gates are
unchanged. Terminal convergence means the final numerical update reached its
tolerance, not that every earlier coarse stage converged or that the pose is
physically correct. Runs that still hit the limit remain in the tables.
Exact ties at the trimming quantile are included so zero-error matches are not
discarded. The new 30-step controls must reproduce the archived matrices and
metrics; verification fails if they do not.

Identical geometric fits are reused within each seed across variants that change
only semantic ranking. Reuse is explicit in each candidate's JSON. Runtime excludes
cached work and historical neural inference and includes first-use normal-cache
work; do not use it as an end-to-end speed comparison.

The [30-step archive](../results/ablation_30_iterations/ARCHIVE.md) preserves the
previous results and identifies their original code revision. It remains a required
comparison input, together with the historical baseline JSON. Neither its poses nor
the historical baseline pose is supplied to fitting; only their hashes enter the
frozen protocol, and the verifier uses them afterward. This follow-up was motivated
by a previously observed 200-step diagnostic, not by an independent new dataset.

## Rebuilding semantic inputs

Pinned upstream model names, revisions, files and hashes are recorded under
`models/`. Obtain their weights separately under the upstream terms; weights are
not redistributed. The renderer and segmentation adapters are included.

```sh
python scripts/prepare_single_date_semantics.py --out results/segmentation_rebuilt
```

This reconstructs the 0929 preprocessing: drone SegFormer predictions, a
Gaussian-rendered source image for the LoveDA building expert, LAS context
projection, and protected-class fusion. The stored projection normals are frozen
0929 preprocessing parameters, not fitted cross-cloud alignments. The original
cropping bounds were estimated before the current ablation; this historical
choice is disclosed rather than described as new blind preprocessing.

Keep rebuilt outputs separate. Check their arrays and hashes against the frozen
inputs before substituting them for reproduction. Different hardware/library
versions may change neural floating-point outputs. A changed cache defines a new
experiment and requires a newly frozen protocol.

## Viewer baseline and BIM

With source data restored, the existing single-date baseline can be rebuilt:

```sh
python scripts/align_0929.py
python scripts/export_3dgs.py
python scripts/prepare_bim.py
python scripts/report_bim.py
```

These commands overwrite their historical output paths. Preserve existing outputs
first if byte-for-byte reproduction matters. The committed scene manifest and
fixed-region diagnostic manifest describe the historical export. The new ablation
stores transforms separately and does not export replacements for the viewer.
Updating the baseline requires updating these viewer manifests as well; do not
present old metrics beside a newly changed export.

## Photo to camera pose

```sh
python scripts/run_camera_localization.py
python scripts/report_camera_localization.py
python scripts/localize_photo.py --image path/to/query.jpg \
  --calibration path/to/calibration.json --output results/camera_localization/new_pose.json
```

The first command creates reference features and reruns the 12-query experiment.
It requires original photographs, COLMAP calibration/reference poses, LAS, and the
historical baseline's COLMAP-to-LAS transform. The overlay report also needs the
BIM mesh. The single-photo command requires the resulting reference cache and a
new output path. The provided example calibration is specific to one camera/image;
do not apply it blindly to another lens or image geometry.

The query pose is not supplied to estimation. Reference poses are supplied.
The underlying LAS/COLMAP reconstruction may contain query photographs, so the
camera study remains an internal-consistency test.
