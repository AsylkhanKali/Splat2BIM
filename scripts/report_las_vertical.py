"""Write the cross-structure LAS diagnostic from saved numerical results."""
import json
from check_las_vertical import OUT


def main():
    data=json.loads((OUT/'audit.json').read_text())
    rows=data['cross_validation'];summary=data['summary'];sensitivity=data['sensitivity']
    failed=[r['date'] for r in summary if not r['roof_shift_improves_bank_blocks']]
    lines=['# Can a single elevation correction align the dates?','',
        '27 October 2026. The original LAS from all seven dates is checked in two previously fixed control volumes: a roof and a narrow riverside structure. Cranes and equipment access routes lie outside these volumes.','',
        f"**A roof-derived correction worsens the riverside control on {', '.join(failed)}.** The result holds for all nine tested XY-radius/block-size combinations. The median decreases on the other three dates, but that does not establish correction of an entire capture. A single vertical correction per cloud is not justified by these two objects.",'',
        'Trial shifts affect temporary arrays in memory only. Original LAS, 3DGS → LAS matrices, exported clouds, and BIM placement are unchanged. 0929 is a comparison date, **not survey ground truth**.','',
        'Follow-up: [rotation/translation with a separate southern control](../las_rigid/REPORT.md), an archived experiment.','',
        '## What the check means','',
        'If a uniform vertical shift were the only discrepancy, a roof-derived correction should also help another persistent structure. We therefore estimate on one object and test on the other, then swap roles. This tests transfer between objects, not independent survey accuracy.','',
        'A positive correction means the source points would be raised toward LAS 0929. The residual is the median absolute height difference across spatial blocks; lower is better.','',
        '| Date | Roof correction, m | Riverside correction, m | Riverside residual before → after roof correction, m | Riverside result |',
        '|---|---:|---:|---:|---|']
    for item in summary:
        row=next(r for r in rows if r['date']==item['date'] and r['fit_region']=='roof')
        result='median decreased' if item['roof_shift_improves_bank_blocks'] else '**worse**'
        lines.append(f"| {item['date']} | {item['roof_delta_z_m']:+.3f} | {item['bank_delta_z_m']:+.3f} | {row['check_block_median_before_m']:.3f} → {row['check_block_median_after_m']:.3f} | {result} |")
    lines+=['| 0929 | — | — | — | Comparison date; not evaluated against itself |','',
        '0504 is particularly informative: the roof suggests +0.87 m and the riverside structure about +0.02 m. Applying the roof correction increases the riverside residual from 0.26 to 0.85 m. On 0609 the median improves slightly, but the height-residual P90 increases from 0.82 to 0.87 m. Improving one median does not imply better overall geometry.','',
        '![Shift estimates and validation on the second object](cross_validation.png)','',
        'Whiskers on the first plot show the interquartile range across blocks, **not a confidence interval**. The roof has a relatively narrow spread; the riverside structure has a much wider spread.','',
        '## Reverse check and 3D distances','',
        '| Date | Riverside correction: roof residual before → after, m | Roof correction: symmetric riverside 3D median before → after, m |',
        '|---|---:|---:|']
    for item in summary:
        a=next(r for r in rows if r['date']==item['date'] and r['fit_region']=='roof')
        b=next(r for r in rows if r['date']==item['date'] and r['fit_region']=='bank_mid')
        lines.append(f"| {item['date']} | {b['check_block_median_before_m']:.3f} → {b['check_block_median_after_m']:.3f} | {a['check_symmetric_3d_before_m']:.3f} → {a['check_symmetric_3d_after_m']:.3f} |")
    lines+=['','The riverside-derived shift reduces the roof median on all six dates, but the improvement on 0504 is only about 2 cm and the residual remains about 86 cm. This does not establish a universal correction.','',
        'The additional check uses the first original point in each 0.25 m voxel. Samples are fixed before trial shifts. Symmetric 3D median means the average of two nearest-neighbor medians: source → 0929 and 0929 → source. Its direction of change agrees with the height check on all six dates. This is agreement between metrics on the same objects, not a new independent observation set. Directional P90 values are retained separately in [cross_validation.csv](cross_validation.csv).','',
        '## Parameter sensitivity','',
        'All combinations of XY radius 0.20 / 0.35 / 0.50 m and block size 1 / 2 / 4 m were tested. All variants are retained; none was selected as best. The table gives the range of riverside median changes after the roof correction; positive means worse.','',
        '| Date | Residual change, m: minimum … maximum | Worse, variants out of 9 |',
        '|---|---:|---:|']
    for item in summary:
        trial=[r['change_in_check_median_m'] for r in sensitivity if r['date']==item['date'] and r['fit_region']=='roof']
        lines.append(f"| {item['date']} | {min(trial):+.3f} … {max(trial):+.3f} | {sum(v>0 for v in trial)} / {len(trial)} |")
    lines+=['','[Full sensitivity results](sensitivity.csv). This establishes robustness to these parameters but does not resolve uncertainty about physical-surface selection.','',
        '## All pairs of seven dates','',
        'All 21 unique date pairs were checked on both objects, giving 42 estimates. Rows are source dates and columns are comparison dates; the reverse half of each matrix is obtained by changing the sign. Diagonal zeros are defined identities, not accuracy tests.','',
        '![Roof/riverside shifts and estimate differences for every date pair](pairwise_offsets.png)','',
        'The two estimates also differ substantially in pairs without 0929: for 0504 → 0708 the roof gives about +0.31 m and the riverside structure about −0.73 m. The disagreement is therefore not solely due to choosing 0929 as reference. See [values and support for each pair](pairwise_offsets.csv).','',
        '## Reproducible protocol','',
        '1. Read the original LAS. Decode coordinates as integer XYZ × scale + offset, then subtract the previous local origin [825800, 842100, 0]. Manual decoding is checked against laspy on 1000 records per date.','2. Apply the identical polygons and height ranges in [regions.json](../static_structures/regions.json) to every date. Masks are not recomputed after trial shifts. Z bounds select the object; there is no error-magnitude filtering.','3. Match mutual nearest neighbors in XY only, within 0.35 m. There is no threshold on Z difference or 3D error.','4. Compute reference Z minus source Z for each pair. Define 2 × 2 m blocks using pair midpoints. Each block needs at least three pairs; the shift is the median of block medians with equal block weights. At least five blocks are required; all 42 estimates meet this threshold.','5. For six dates relative to 0929, test the correction estimated on one object on blocks of the other, without refitting. Control pairs and 3D samples are identical before and after the shift.','',
        'The baseline protocol has 90–100 eligible roof blocks and 19–22 riverside blocks per date pair. Total XY-pair counts, mutual-match fractions, and eligible-block counts are saved separately: not every pair falls in a block with three neighbors. NPZ contains all mutual pairs; `matched_blocks.csv` contains only eligible blocks.','',
        '## LAS header findings','',
        'All seven LAS 1.2 files identify `CaptureReality` as generator; both georeferencing VLRs are byte-identical by SHA-256. Different scale/offset values are correctly decoded. Z quantization is approximately 1.3–1.9 × 10⁻⁸ m, so storage rounding cannot explain discrepancies of tens of centimeters.','',
        'Identical metadata does not prove shared actual georeferencing or correct elevations. The headers do not confirm a common vertical datum. LAS format does not make these independent laser or survey measurements. [audit.json](audit.json) preserves header fields, GeoKeys, VLR hashes, and input hashes. All seven LAS hashes match the earlier manifests; 21 registration files remained unchanged during this check.','',
        '## Limitations and justified next steps','',
        'XY proximity does not guarantee the same physical surface. Visible sides, coverage, and vegetation may differ on the narrow riverside object; the 0929 roof volume includes lower points. A persistent outline does not prove immobility. This check does not determine horizontal shift, tilt, scale, or spatial deformation.','',
        'The conclusion is limited to the tested hypothesis: a single roof-derived height correction cannot confidently remove inter-date discrepancies. This neither proves a specific cause nor measures absolute BIM accuracy. Physical correction needs additional distributed support surfaces or identified control points, with georeferencing and vertical-datum checks. BIM should not be fitted separately to each date merely to improve visual overlap.','',
        'The previous [3DGS → LAS stable-structure evaluation](../static_structures/REPORT.md) remains separate: agreement with same-date LAS and agreement between LAS dates are different questions.','',
        '## Repeat the calculation','',
        '```sh','OPENBLAS_NUM_THREADS=3 OMP_NUM_THREADS=3 .venv/bin/python scripts/check_las_vertical.py',
        '.venv/bin/python scripts/report_las_vertical.py',
        'OPENBLAS_NUM_THREADS=3 OMP_NUM_THREADS=3 .venv/bin/python -m unittest discover -s scripts -p "test_*.py" -v','```','',
        'Numerical tests check the sign of a known vertical shift and its reverse pair, failure to transfer an incorrect correction to another object, absence of false zero errors with insufficient overlap, and absence of Z-error filtering. The full suite at this experiment stage contained 16 tests.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print(OUT/'REPORT.md')


if __name__=='__main__':main()
