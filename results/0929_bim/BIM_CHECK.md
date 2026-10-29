# 0929 ↔ BIM inspection

3DGS and all three BIM revisions share one local metre frame. BIM receives only the same origin shift `[825800, 842100, 0]`; no additional BIM fit is applied. The common placement appears consistent, but this is preliminary inspection, not survey acceptance.

## Revisions and completeness

| Revision | Elements converted | Not represented by a mesh |
|---|---:|---:|
| v1 | 5679 | 1 |
| v2 | 5579 | 1 |
| v3 | 5586 | 1 |

Only `IFC Project Base Point`, an annotation rather than a construction surface, is excluded in all three files.
v3 retains geometry for all 5579 v2 elements and adds 7: four kiosks and three unnamed elements. It is used for inspection as the more complete of these two revisions; its official status requires confirmation.
v1 and v3 share 5479 elements with identical GUIDs and geometry; 200 v1 elements are absent from v3 and 107 are added. Revisions should not be merged into one model.

## BIM → LAS distances

Each element has 500 surface samples drawn with probability proportional to triangle area. The table reports the median of individual element medians, not global registration accuracy.

| Group | Elements | Median of medians, m | 90th percentile of medians, m |
|---|---:|---:|---:|
| Deck | 80 | 0.721 | 1.100 |
| Canopy | 60 | 0.366 | 0.487 |
| Pier | 15 | 0.706 | 2.420 |
| Pile Cap | 4 | 3.792 | 4.236 |
| Kiosk | 4 | 0.905 | 1.036 |
| Underground piles | 69 | 18.845 | 21.065 |

## Visual inspection areas

| v3 element | Median distance to LAS, m | Fraction of samples within 1 m |
|---|---:|---:|
| Deck near FP4/FP5 · #9525 | 0.224 | 100.0% |
| Canopy, north section · #49665 | 0.151 | 100.0% |
| FP1 support · #11021 | 0.344 | 87.0% |
| Pile cap: check visibility · #10912 | 2.638 | 5.6% |
| Kiosk, west section · #5828 | 0.406 | 93.8% |

These areas were selected for inspection, not as independent control samples. Full element-level results are in `v3_element_distances.csv`.
Pile caps and piles include hidden and underground surfaces. Their large distances to aerial observations do not automatically indicate model displacement or missing structures. Decks also include underside and side surfaces that are not visible from every viewpoint.

## How to open

Open the running viewer: http://127.0.0.1:8765/viewer/
Toggle 3DGS, LAS, and BIM revisions; the structure list focuses selected elements and outlines them in cyan. The wireframe intentionally includes hidden parts: it is an overlay aid, not physically exact visibility through translucent Gaussians. Disable BIM for an unobstructed capture view.

To restart from the project root:

```sh
npm install --prefix viewer --ignore-scripts --no-audit --no-fund
.venv/bin/python scripts/prepare_bim.py
.venv/bin/python scripts/report_bim.py
.venv/bin/python scripts/serve_viewer.py
```

The viewer uses locally installed Spark 2.4.0 and Three.js 0.180.0. The server binds only to 127.0.0.1 and serves only `viewer/` and `results/`; raw input directories are not served.
