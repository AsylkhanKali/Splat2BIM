# Shared-region checks across all dates

Shared cells: 21, each 20 × 20 m. Every date/cell has at least 50 held-out 3DGS centers inside its LAS XY coverage. The grid is fixed in the common local frame. All seven registration matrices remain unchanged.

This compares shared plan-view regions, not necessarily the same physical surfaces. Construction and visibility inside a cell change over time.

| Date | Control centers | Median of cell medians, m | Median cell P90, m | Mean cell fraction < 1 m | Retained from previous evaluation |
|---|---:|---:|---:|---:|---:|
| 0331 | 7,528 | 0.257 | 0.737 | 86.8% | 17.0% |
| 0504 | 8,165 | 0.199 | 0.635 | 91.1% | 16.1% |
| 0609 | 7,364 | 0.274 | 0.717 | 90.2% | 15.2% |
| 0708 | 4,563 | 0.236 | 0.985 | 86.2% | 13.0% |
| 0723 | 6,744 | 0.213 | 0.765 | 93.0% | 17.9% |
| 0831 | 7,585 | 0.188 | 0.512 | 95.9% | 16.4% |
| 0929 | 4,812 | 0.212 | 0.718 | 92.8% | 10.4% |

Every cell has equal weight. The median of cell medians and the median of cell P90 values are not the pooled median/P90 of all points. Conventional point-weighted metrics are also retained in comparison.csv.

## Inspection regions

The eight cells below have the largest P90 on at least one date. They are a shortlist for inspecting problematic areas, not a representative site-wide sample. The aggregate table above uses all shared cells.

| Zone | Local XY of lower corner, m | Date with highest P90 | P90, m |
|---|---|---|---:|
| 1 | 0, -60 | 0331 | 44.494 |
| 2 | 0, -40 | 0331 | 43.024 |
| 3 | 20, -80 | 0609 | 16.433 |
| 4 | 20, 100 | 0504 | 14.885 |
| 5 | -40, 80 | 0609 | 7.707 |
| 6 | 0, 100 | 0708 | 3.685 |
| 7 | -20, -20 | 0708 | 2.007 |
| 8 | -40, 60 | 0708 | 1.998 |

## What was checked

Distances from held-out centers to the full LAS were recomputed. Counts and four original metrics were reproduced for each date within 1e-8. The previous spatial fit exclusions are retained; their physical size differs by date. Cells qualify by having at least 50 points, without selection by distance magnitude.

A large distance may reflect local mismatch, noise, a Gaussian center off the physical surface, or incomplete LAS visibility. This check alone does not identify the cause, measure structural movement, or establish survey accuracy. A small distance on a plane also does not rule out along-plane sliding.

## Files and viewing

Crane inspection in zones 1–2, height comparisons with the full LAS, and viewer screenshots are retained in the [27 October inspection notes](INSPECTION.md).

In the viewer, expand Alignment checks, select Shared 20 × 20 m cells, enable colored control points, and choose a zone. Point colors show distance to LAS; common_area_map.png shows each cell P90. Points are deliberately drawn over scene geometry for diagnosis, including hidden points.

comparison.csv contains aggregate metrics; cells.csv contains all shared cells; *_samples.npz stores coordinates and exact distances; *_distances.ply stores colored control centers. Original scenes and registration results are not overwritten.

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python scripts/check_common_areas.py
```
