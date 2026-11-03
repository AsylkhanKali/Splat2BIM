# 0929 controlled registration ablation — convergence update

The earlier 30-step runs were iteration-limited: their mean median was 1.350 m. With up to 500 steps per gate and unchanged stopping tolerances, the geometric primary branch reaches 0.220 m; 2/3 runs reach the terminal update tolerance. This correction concerns optimization completion, not a relaxed evaluation threshold. 1 of 3 primary runs remain iteration-limited and explicitly flagged. A larger cap alone does not guarantee convergence.

Across three sampling seeds, mean held-out median changes from 2.278 m after PCA to 0.220 m after ICP. Adding semantic ranking changes that value by +0.000000 m. The geometric and semantic primary branches selected identical final matrices on all three seeds: semantic ranking did not change the selected alignment in this experiment.

| Method | Mean median, m | Mean P90, m | Mean RMSE, m | Terminal convergence |
|---|---:|---:|---:|---:|
| PCA only | 2.278 | 14.703 | 15.321 | N/A |
| PCA + ICP | 0.220 | 0.885 | 15.511 | 2/3 |
| PCA + semantics + ICP | 0.220 | 0.885 | 15.511 | 2/3 |

The historical RTK/COLMAP-assisted transform gives median 0.223 m on the same 48,081 points. It is a comparator, not an input to the new fits or independent ground truth. Large tail distances remain in RMSE.

[Full HTML report](inspection.html) · [Every run](summary.csv) · [Protocol](protocol.json) · [Iteration comparison](iteration_comparison.csv) · [Archived 30-step report](../ablation_30_iterations/inspection.html)

Only one previously explored date is evaluated. These are reference-agreement measurements, not independent survey accuracy.
