# Supervisor review: controlled 0929 study

The current deliverable is a reproducible research prototype. The controlled
study has 48 comparisons (16 settings, three sampling seeds), fixed evaluation
indices, saved transformation matrices, convergence diagnostics and an HTML
report. The seeds are not independent captures.

## Results with a 200-iteration cap

The cap is **200 updates per distance gate**, not 200 updates for the entire
pipeline. There are four point-to-point and five point-to-plane gates. Early
stopping is enabled. Metrics below average the per-run metrics over seeds
17, 29 and 43, on the same 48,081 held-out Gaussian centers.

| Measurement | Result |
|---|---:|
| Mean median distance to nearest LAS point | 0.220495 m (22.05 cm) |
| Range of per-run medians | 0.220376–0.220635 m |
| Mean P90 distance | 0.884592 m (88.46 cm) |
| Mean fraction within 0.5 m | 81.53% |
| Mean fraction within 1 m | 91.29% |
| Mean full-cohort RMSE | 15.510971 m |
| Runs reaching the strict terminal tolerance | 2 / 3 |

The 30-step mean median was 1.350226 m; the 200-step value is about 83.7% lower.
The 500-step result is practically unchanged. Seed 43 remains iteration-limited
at both 200 and 500; no tolerance was relaxed and no failed run was removed.
The large RMSE comes from the retained error tail. These are distances from
Gaussian centers to LAS, not a 22 cm accuracy guarantee for all objects or
independent survey validation. BIM is not the evaluation target.

Semantic labels rank initialization candidates; they do not constrain the ICP
point pairs. The matched primary geometry and semantic-ranking branches select
identical final matrices. The 200-step control is the geometric branch, not a
separate 200-step semantic experiment.

## Inspect the evidence

1. Read `results/ablation/inspection.html` for the current comparison, pipeline,
   settings, convergence flags, uncertainty estimates and matrix downloads.
2. Inspect `results/ablation/iteration_comparison.csv` and
   `results/ablation/iteration_budget_200/{17,29,43}/result.json` for exact values.
3. Open the viewer to inspect the supplied scene and BIM. Its full-scene export
   remains the historical RTK/COLMAP-assisted baseline. It is not the PCA-only
   experiment's rendered output.
4. Inspect `results/camera_localization/inspection.html` separately. It reports
   a calibrated, same-site 12-query experiment, not arbitrary-photo localization.

## Run and verify

A Git clone contains compact evidence, not the private inputs. A separately
supplied lab package can include those inputs and installed browser dependencies.
The HTML reports can be opened directly. To use the 3D viewer, run from the
project root:

```sh
python scripts/serve_viewer.py
```

Open `http://127.0.0.1:8765/viewer/`. Use `python3` if that is your Python command,
or `py -3.11` on Windows. Stop any existing server on port 8765 first. If browser
dependencies were not supplied, install Node/npm and run `npm ci --prefix viewer`.

For numerical checks, create a Python 3.11 environment and install the review
dependencies (the recorded development platform is macOS ARM64):

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-review.txt
python -m unittest discover -s scripts -p 'test_*.py' -v
```

On Windows create the environment with `py -3.11 -m venv .venv` and activate it
with `.venv\Scripts\Activate.ps1`. Other platforms have not been validated end to
end; exact wheel availability and floating-point reproducibility may differ.

After restoring the raw inputs and frozen labels, independently check the saved
48 comparisons without repeating the full optimization:

```sh
python scripts/verify_registration_ablation.py --out results/ablation
```

The verifier rewrites its diagnostic receipts, not the fitted matrices. Preserve
the supplied package before running experiments. To refit every configuration
and produce a new report in a fresh directory:

```sh
python scripts/run_registration_ablation.py --out results/ablation_supervisor
python scripts/verify_registration_ablation.py --out results/ablation_supervisor
python scripts/report_registration_ablation.py --out results/ablation_supervisor
```

Set `OPENBLAS_NUM_THREADS=2` and `OMP_NUM_THREADS=2` to limit CPU parallelism.
The full run is substantially longer than the synthetic tests. Frozen semantic
labels are sufficient for this experiment; rerunning neural segmentation needs
the recorded model weights and the full dependencies in `requirements.txt`.
See [REPRODUCING.md](REPRODUCING.md) for semantic, BIM and camera reconstruction.
