"""Audit held-out Gaussian centres on the same metric grid for all seven dates."""
import csv
import hashlib
import json

import laspy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm,ListedColormap
from matplotlib.patches import Rectangle
import numpy as np
from plyfile import PlyData,PlyElement
from scipy.spatial import cKDTree

from align_dates import ROOT,ORIGIN,DATES,samples
from align_0929 import apply,summary

OUT=ROOT/'results/common_areas'
CELL=20.
MIN_COUNT=50
COLORS=['#2dd4bf','#fde047','#fb923c','#f472b6']
LIMITS=[.25,.5,1.]


def grid_indices(points,cell=CELL):
    return np.floor(points[:,:2]/cell).astype(np.int64)


def summarize_cells(points,errors,cell=CELL):
    indices=grid_indices(points,cell)
    unique,inverse=np.unique(indices,axis=0,return_inverse=True)
    rows={}
    for i,(x,y) in enumerate(unique):
        selected=inverse==i;z=points[selected,2];e=errors[selected]
        rows[f'{x},{y}']={'ix':int(x),'iy':int(y),'count':int(selected.sum()),
            'median_m':float(np.median(e)),'p90_m':float(np.quantile(e,.9)),
            'within_1m':float(np.mean(e<1.)),
            'z_median':float(np.median(z)),'z05':float(np.quantile(z,.05)),'z95':float(np.quantile(z,.95))}
    return rows


def common_cells(by_date,min_count=MIN_COUNT):
    eligible=[{key for key,row in rows.items() if row['count']>=min_count} for rows in by_date.values()]
    return sorted(set.intersection(*eligible),key=lambda key:tuple(map(int,key.split(','))))


def balanced_metrics(rows,keys):
    return {'cells':len(keys),'samples':sum(rows[k]['count'] for k in keys),
        'median_of_cell_medians_m':float(np.median([rows[k]['median_m'] for k in keys])),
        'median_of_cell_p90_m':float(np.median([rows[k]['p90_m'] for k in keys])),
        'mean_cell_fraction_within_1m':float(np.mean([rows[k]['within_1m'] for k in keys]))}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    clouds={};cells={};checks={};inputs=[]
    for date in DATES:
        base=ROOT/'results/0929_baseline' if date=='0929' else ROOT/f'results/multidate/{date}'
        path=base/'alignment.json';info=json.loads(path.read_text())
        assert info.get('status')!='rejected'
        np.testing.assert_array_equal(info['origin_global'],ORIGIN)
        source,_,held=samples(date);points=apply(source[held],np.array(info['ply_to_las_local']))
        las=laspy.read(ROOT/f'assets/{date}/{date}.las');reference=np.c_[las.x,las.y,las.z]-ORIGIN
        errors=cKDTree(reference).query(points,workers=4)[0]
        overlap=cKDTree(reference[:,:2]).query(points[:,:2],workers=4)[0]<1.
        points,errors=points[overlap],errors[overlap]
        old=info['ply_withheld_overlap_distances_m'];actual=summary(errors)
        assert len(errors)==old['count']
        np.testing.assert_allclose([actual[k] for k in ('median','p90','p95','rmse')],
                                  [old[k] for k in ('median','p90','p95','rmse')],atol=1e-8,rtol=1e-8)
        clouds[date]=(points,errors);cells[date]=summarize_cells(points,errors)
        checks[date]={'previous_metrics_reproduced':True,'heldout_overlap_count':len(points)}
        inputs.append({'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        print(date,'held-out metrics reproduced; occupied cells',len(cells[date]),flush=True)
    common=common_cells(cells)
    if not common:raise ValueError('No common cells meet the declared sample threshold')
    print('Common cells',len(common),flush=True)
    common_set=set(common);rows=[];regions=[]
    # Inspection shortlist only. The common-area summary always uses every eligible cell.
    ranked=sorted(common,key=lambda k:max(cells[d][k]['p90_m'] for d in DATES),reverse=True)
    for i,key in enumerate(ranked[:8]):
        ix,iy=map(int,key.split(','));worst=max(DATES,key=lambda d:cells[d][key]['p90_m'])
        regions.append({'id':key,'label':f'Зона {i+1} · {ix*CELL:.0f}, {iy*CELL:.0f} м',
            'worst_date':worst,'worst_p90_m':cells[worst][key]['p90_m'],
            'bounds_xy':[ix*CELL,iy*CELL,(ix+1)*CELL,(iy+1)*CELL],
            'dates':{d:cells[d][key] for d in DATES}})
    manifest={'origin_global':ORIGIN.tolist(),'cell_size_m':CELL,'min_samples_per_date':MIN_COUNT,
        'common_cells':common,'common_cell_count':len(common),'color_limits_m':LIMITS,'colors':COLORS,
        'regions':regions,'dates':{}}
    fig,axes=plt.subplots(2,4,figsize=(16,10),constrained_layout=True)
    cmap=ListedColormap(COLORS);norm=BoundaryNorm([0,.25,.5,1,1000000],4)
    union=set().union(*(set(cells[d]) for d in DATES))
    x0=min(cells[d][k]['ix'] for d in DATES for k in common)*CELL
    x1=(max(cells[d][k]['ix'] for d in DATES for k in common)+1)*CELL
    y0=min(cells[d][k]['iy'] for d in DATES for k in common)*CELL
    y1=(max(cells[d][k]['iy'] for d in DATES for k in common)+1)*CELL
    palette=np.array([[int(c[j:j+2],16) for j in (1,3,5)] for c in COLORS],dtype='u1')
    for date,ax in zip(DATES,axes.flat):
        points,errors=clouds[date];indices=grid_indices(points)
        keep=np.array([f'{x},{y}' in common_set for x,y in indices])
        p,e=points[keep],errors[keep]
        record={'date':date,**balanced_metrics(cells[date],common),
            'point_weighted_median_m':float(np.median(e)),'point_weighted_p90_m':float(np.quantile(e,.9)),
            'retained_fraction_of_previous_holdout':float(keep.mean())}
        rows.append(record)
        vertices=np.empty(len(p),dtype=[(k,'f4') for k in ('x','y','z')]+[(k,'u1') for k in ('red','green','blue')])
        for i,k in enumerate(('x','y','z')):vertices[k]=p[:,i]
        colors=palette[np.searchsorted(LIMITS,e,side='right')]
        for i,k in enumerate(('red','green','blue')):vertices[k]=colors[:,i]
        output=OUT/f'{date}_distances.ply'
        PlyData([PlyElement.describe(vertices,'vertex')],text=False,comments=['Held-out centres in common 20m cells; color is distance to own LAS, not survey error.']).write(output)
        saved=PlyData.read(output,mmap='r')['vertex'].data
        assert len(saved)==len(p) and np.isfinite(np.column_stack([saved[k] for k in ('x','y','z')])).all()
        np.testing.assert_array_equal(np.column_stack([saved[k] for k in ('red','green','blue')]),colors)
        np.savez_compressed(OUT/f'{date}_samples.npz',points=p,distance_m=e,grid_xy=indices[keep])
        manifest['dates'][date]={**record,'points_url':'/'+str(output.relative_to(ROOT)),
            'cells':{key:cells[date][key] for key in common}}
        for key in union:
            x,y=map(int,key.split(','))
            ax.add_patch(Rectangle((x*CELL,y*CELL),CELL,CELL,facecolor='#e8edf0',edgecolor='white',linewidth=.3))
        for key in common:
            r=cells[date][key]
            ax.add_patch(Rectangle((r['ix']*CELL,r['iy']*CELL),CELL,CELL,facecolor=cmap(norm(r['p90_m'])),edgecolor='white',linewidth=.4))
        for i,key in enumerate(ranked[:3]):
            r=cells[date][key];ax.text((r['ix']+.5)*CELL,(r['iy']+.5)*CELL,str(i+1),ha='center',va='center',fontsize=7)
        ax.set(xlim=(x0-10,x1+10),ylim=(y0-10,y1+10),aspect='equal',title=f"{date} · cell P90",xlabel='Local X, m',ylabel='Local Y, m')
    ax=axes.flat[-1];ax.axis('off')
    ax.text(0,1,f'{len(common)} common cells\n20 × 20 m; ≥50 held-out centres/date\n\nEach map colors cell P90.\nGray: outside common sample support.\nNumbers: top 3 inspection zones.\n\nOne unchanged transform per date.\nDistances are to that date’s own LAS,\nnot construction change or survey accuracy.',va='top',fontsize=12,linespacing=1.6)
    for i,(color,label) in enumerate(zip(COLORS,['< 0.25 m','0.25–0.50 m','0.50–1.00 m','≥ 1.00 m'])):
        ax.add_patch(Rectangle((0,.22-i*.055),.055,.035,transform=ax.transAxes,facecolor=color))
        ax.text(.08,.237-i*.055,label,transform=ax.transAxes,va='center',fontsize=11)
    fig.savefig(OUT/'common_area_map.png',dpi=150);plt.close(fig)
    with (OUT/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (OUT/'cells.csv').open('w',newline='') as f:
        fields=['date','cell_id','ix','iy','count','median_m','p90_m','within_1m','z_median','z05','z95']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for date in DATES:
            for key in common:writer.writerow({'date':date,'cell_id':key,**cells[date][key]})
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (OUT/'validation.json').write_text(json.dumps({'previous_metrics':checks,'common_cell_count':len(common),'minimum_count_observed':min(cells[d][k]['count'] for d in DATES for k in common),'alignment_inputs':inputs,'colored_exports_reread':7},indent=2)+'\n')
    lines=['# Проверка одних и тех же участков во всех датах','',
        f'Общие ячейки: {len(common)}; размер каждой — 20 × 20 м. В каждой ячейке и каждой дате есть не меньше 50 отложенных центров 3DGS в XY-покрытии своего LAS. Сетка закреплена в общей локальной системе координат. Все семь матриц совмещения оставлены без изменений.','',
        'Это сравнение общих участков в плане, а не одних и тех же физических поверхностей. Стройка и видимость объектов внутри ячейки меняются со временем.','',
        '| Дата | Контрольных центров | Медиана медиан ячеек, м | Медиана P90 ячеек, м | Средняя по ячейкам доля < 1 м | Сохранено от прежней проверки |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['date']} | {r['samples']:,} | {r['median_of_cell_medians_m']:.3f} | {r['median_of_cell_p90_m']:.3f} | {r['mean_cell_fraction_within_1m']:.1%} | {r['retained_fraction_of_previous_holdout']:.1%} |")
    lines+=['','Каждая ячейка имеет одинаковый вес. Медиана медиан и медиана P90 ячеек не равны общей медиане или P90 всех точек. Обычные показатели с весом по числу точек также сохранены в comparison.csv.','',
        '## Участки для осмотра','',
        'Ниже перечислены восемь ячеек с наибольшим P90 хотя бы в одной дате. Это выборка проблемных зон для осмотра, а не репрезентативная оценка всей площадки. Общая таблица выше использует все общие ячейки.','',
        '| Зона | Локальные XY нижнего угла, м | Дата наибольшего P90 | P90, м |','|---|---|---|---:|']
    for i,r in enumerate(regions):lines.append(f"| {i+1} | {r['bounds_xy'][0]:.0f}, {r['bounds_xy'][1]:.0f} | {r['worst_date']} | {r['worst_p90_m']:.3f} |")
    lines+=['','## Что именно проверено','',
        'Заново вычислены расстояния от отложенных центров до полного LAS. Количество точек и четыре исходных показателя каждой даты воспроизведены с допуском 1e-8. Для контроля используются прежние пространственные исключения из подгонки; их физический размер различается между датами. Ячейки включаются по наличию минимум 50 точек, без отбора по величине расстояния.','',
        'Большое расстояние может означать локальное несовпадение, шум, центр гауссиана вне поверхности или недостаточную видимость поверхности в LAS. Эта проверка сама по себе не устанавливает причину, не измеряет перемещение конструкции и не подтверждает геодезическую точность. Небольшое расстояние на плоском участке также не исключает сдвиг вдоль плоскости.','',
        '## Файлы и просмотр','',
        'Осмотр крана в зонах 1–2, сравнение высот с полным LAS и снимки просмотрщика сохранены в [заметках проверки от 27 октября](INSPECTION.md).','',
        'В просмотрщике раскройте «Проверка совмещения», выберите «Общие ячейки 20 × 20 м», включите цветные контрольные точки и выберите зону. Цвет отдельных точек показывает их расстояние до LAS; карта common_area_map.png показывает P90 каждой ячейки. Точки в 3D намеренно рисуются поверх сцены для диагностики, включая скрытые точки.','',
        'comparison.csv содержит итоговые числа, cells.csv содержит все общие ячейки, *_samples.npz хранит координаты и точные расстояния, *_distances.ply содержит цветные контрольные центры. Исходные сцены и результаты регистрации не перезаписываются.','',
        '```sh','OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python scripts/check_common_areas.py','```']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':main()
