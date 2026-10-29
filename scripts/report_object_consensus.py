"""Report whole-scene distances, preserved fragment identities and fixed structures."""
import json

import laspy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

from align_dates import ROOT,ORIGIN,DATES
from align_0929 import apply,summary
from align_object_consensus import OUT
from check_static_structures import volume_mask
from check_las_vertical import write_csv,file_hash


def plot_pairs(folder,audit,target,color):
    run=next(r for r in audit['runs'] if r['method']=='multi_fragment')
    if 'matrix_local' not in run:return
    pairs=run['final_spatial_support'] or run['final_pair_checks'][:5]
    source=np.load(folder/'source_candidate_points.npz');dest=np.load(folder/'target_candidate_points.npz')
    source_points=[apply(source[r['source_id']],np.array(run['matrix_local'])) for r in pairs]
    target_points=[dest[r['target_id']] for r in pairs]
    joint=np.concatenate(source_points+target_points);lo=joint[:,:2].min(0)-8;hi=joint[:,:2].max(0)+8
    keep=np.all((target[:,:2]>=lo)&(target[:,:2]<=hi),axis=1)
    background=target[keep];rgb=color[keep];order=np.argsort(background[:,2])[::3]
    fig,axes=plt.subplots(1,2,figsize=(12,6),constrained_layout=True)
    for ax,patches,title in zip(axes,[source_points,target_points],['Фрагменты 3DGS после совмещения','Соответствующие фрагменты LAS']):
        ax.scatter(background[order,0],background[order,1],c=rgb[order],s=1,alpha=.55)
        for i,patch in enumerate(patches):
            shown=patch[::max(1,len(patch)//300)];color_pair=plt.get_cmap('tab10')(i%10)
            ax.scatter(shown[:,0],shown[:,1],s=5,color=color_pair)
            center=patch.mean(0);ax.text(center[0],center[1],str(i+1),color='black',fontsize=10,
                bbox=dict(facecolor='white',edgecolor=color_pair,boxstyle='round,pad=.2'))
        ax.set(aspect='equal',xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),title=title,xlabel='X, м',ylabel='Y, м')
    label=f"{len(run['final_spatial_support'])} разнесённых пар прошли проверку" if run['final_spatial_support'] else 'соответствия не подтверждены'
    fig.suptitle(f"{audit['date']} · {label}")
    fig.savefig(folder/'correspondences.png',dpi=150);plt.close(fig)


def main():
    audits={d:json.loads((OUT/d/'audit.json').read_text()) for d in DATES}
    regions_path=ROOT/'results/static_structures/regions.json'
    regions=json.loads(regions_path.read_text())['regions'];rows=[];static=[];pair_rows=[]
    for date,audit in audits.items():
        folder=OUT/date;rows.extend(audit['comparison'])
        las=laspy.read(ROOT/f'assets/{date}/{date}.las');target=np.c_[las.x,las.y,las.z]-ORIGIN
        color=np.c_[las.red,las.green,las.blue].astype(float);color/=65535 if color.max()>255 else 255
        base=ROOT/('results/0929_baseline' if date=='0929' else f'results/multidate/{date}')
        baseline=np.array(json.loads((base/'alignment.json').read_text())['ply_to_las_local'])
        vertices=PlyData.read(ROOT/f'assets/{date}/{date}.ply',mmap='r')['vertex'].data[5::10]
        points=np.c_[vertices['x'],vertices['y'],vertices['z']].astype(float)
        valid=(vertices['opacity']>0)&np.isfinite(points).all(1);points=points[valid]
        held=np.floor(points/.4).sum(1).astype(int)%4==0
        original=apply(points,baseline)
        matrices={r['method']:np.array(r['matrix_local']) for r in audit['runs'] if 'matrix_local' in r}
        matrices['existing_baseline']=baseline
        controls={}
        for region in regions:
            region_mask=volume_mask(original,region);reference=target[volume_mask(target,region)]
            for scope,mask in [('unseen_records',np.ones(len(points),bool)),('spatial_holdout',held)]:
                selected=region_mask&mask;controls[region['id']+'_'+scope]=points[selected]
                for method,matrix in matrices.items():
                    distance=cKDTree(reference).query(apply(points[selected],matrix))[0]
                    static.append({'date':date,'region':region['id'],'scope':scope,'method':method,
                        **(summary(distance) if len(distance) else {'count':0,'median':None,'p90':None,'p95':None,'rmse':None}),
                        'adequate_support':len(distance)>=30,'target_count':len(reference)})
        np.savez_compressed(folder/'static_control_source_points.npz',**controls)
        for run in audit['runs']:
            for pair in run.get('final_pair_checks',[]):pair_rows.append({'date':date,'method':run['method'],**pair})
        plot_pairs(folder,audit,target,color)
    write_csv(OUT/'comparison.csv',rows);write_csv(OUT/'static_structures.csv',static);write_csv(OUT/'correspondences.csv',pair_rows)
    def run(date,method):return next(r for r in audits[date]['runs'] if r['method']==method)
    def metric(date,method):return next((r for r in rows if r['date']==date and r['method']==method and r['scope']=='fixed_baseline_coverage'),None)
    def value(row,key='median'):return '—' if row is None or row[key] is None else f"{row[key]:.3f}"
    passed=[d for d in DATES if run(d,'multi_fragment')['status']=='candidate_for_review']
    strict=[d for d in DATES if run(d,'multi_fragment').get('strict_forward_080_spatial_support_count',0)>=3 and d in passed]
    labels={'candidate_for_review':'кандидат для проверки','rejected_no_spatial_consensus':'нет пространственного согласия',
        'rejected_training_alignment':'совмещение отклонено','rejected_object_correspondences':'пары не подтверждены'}
    lines=['# Совмещение по нескольким пространственно согласованным фрагментам','',
        '**Что сделано:** автоматический поиск светлых приподнятых поверхностей дополнен проверкой нескольких соответствий. Новая привязка не использует размеченную крышу, RTK/COLMAP, ручную начальную матрицу или прежние преобразования. Проверены все семь дат.','',
        f"Основной фильтр проходит {len(passed)} из 7 дат: {', '.join(passed) or 'ни одна'}. При дополнительном строгом требовании ≥80% исходного фрагмента ближе 1 м остаются {len(strict)}: {', '.join(strict) or 'ни одна'}. Это кандидаты для проверки, не геодезическая приёмка и не распознавание классов объектов.",'',
        '## Сравнение на одинаковых контрольных точках','',
        'Все расстояния — от центров гауссианов до ближайших точек LAS. Общая область покрытия заранее фиксируется рабочей привязкой. Она не меняется под результат каждого метода. Полные ошибки по всем отложенным точкам также сохранены, включая неперекрывающиеся части.','',
        '| Дата | Один фрагмент: медиана, м | Несколько: медиана / P90, м | Рабочая привязка: медиана, м | Разнесённых пар | Статус нового метода |',
        '|---|---:|---:|---:|---:|---|']
    for date in DATES:
        r=run(date,'multi_fragment');m=metric(date,'multi_fragment')
        lines.append(f"| {date} | {value(metric(date,'single_fragment'))} | {value(m)} / {value(m,'p90')} | {value(metric(date,'existing_baseline'))} | {len(r.get('final_spatial_support',[]))} | {labels[r['status']]} |")
    lines+=['','«Один фрагмент» — контрольный вариант с тем же расширенным набором поверхностей, что у нового метода. В обоих случаях до четырёх различных начальных матриц проходят одинаковый ICP; по обучающей медиане выбирается одна для одинакового окончательного уточнения. Это позволяет отделить вклад пространственного согласия от простого расширения набора кандидатов. Рабочая привязка приведена как ориентир с другими входными условиями.','',
        '## Контроль по неподвижным конструкциям','',
        'Использованы прежние проверенные объёмы крыши стационарного блока и средней части прибрежной конструкции. Краны и дорожная техника находятся вне этих контрольных областей. Проверяются отдельные исходные записи PLY с индексами 5, 15, 25…; они не участвовали в поиске или подгонке, где использовались только чётные индексы. Состав точек зафиксирован рабочей матрицей: после нового преобразования удалённые точки не выбрасываются. Это локальные контрольные области, не семантическая очистка всей площадки.','',
        'Основная таблица ниже — проверка на других записях, которые могут соседствовать с обучающими точками. Она не равнозначна пространственно независимой проверке. Более строгая выборка только из отложенных плиток сохранена отдельно; во многих маленьких областях в ней меньше 30 точек или нет точек. Это ограничение не скрывается увеличением выборки.','',
        '| Дата | Крыша: несколько / рабочая, м | Прибрежная конструкция: несколько / рабочая, м | Контрольных точек: крыша / берег |',
        '|---|---:|---:|---:|']
    for date in DATES:
        def s(region,method):return next((r for r in static if r['date']==date and r['region']==region and r['method']==method and r['scope']=='unseen_records'),None)
        roof=s('roof','multi_fragment');bank=s('bank_mid','multi_fragment')
        rb=s('roof','existing_baseline');bb=s('bank_mid','existing_baseline')
        lines.append(f"| {date} | {value(roof)} / {value(rb)} | {value(bank)} / {value(bb)} | {rb['count']} / {bb['count']} |")
    lines+=['','Количество точек в более строгой пространственной проверке:','',
        '| Дата | Крыша | Берег | Достаточно ≥30 в обеих областях |','|---|---:|---:|---|']
    for date in DATES:
        counts=[next(r['count'] for r in static if r['date']==date and r['region']==region and r['method']=='existing_baseline' and r['scope']=='spatial_holdout') for region in ['roof','bank_mid']]
        lines.append(f"| {date} | {counts[0]} | {counts[1]} | {'да' if min(counts)>=30 else '**нет**'} |")
    lines+=['','## Проверка именно предложенных пар','',
        'Пары сохраняют свои идентификаторы после ICP: алгоритм не может заменить ошибочную пару ближайшей новой. Для каждой требуется ≥50% точек ближе 1 м **в обоих направлениях**, согласованные размеры/нормали и расстояние между центрами ≤5 м. Затем нужны минимум три пары: центры разнесены минимум на 10 м в обеих сценах и не лежат почти на одной прямой. Несколько фрагментов могут всё равно принадлежать одному крупному сооружению; это не доказательство трёх разных зданий.','',
        '| Дата | Разнесённых пар, основной фильтр | При дополнительном пороге 80% | Начальный P75 до ICP: один / несколько, м |',
        '|---|---:|---:|---:|']
    for date in DATES:
        a=run(date,'single_fragment');b=run(date,'multi_fragment')
        def initial(r):return f"{min(c['initial_score_p75_m'] for c in r['coarse_candidates']):.2f}" if 'coarse_candidates' in r else '—'
        lines.append(f"| {date} | {len(b.get('final_spatial_support',[]))} | {b.get('strict_forward_080_spatial_support_count',0)} | {initial(a)} / {initial(b)} |")
    for date in DATES:
        if (OUT/date/'correspondences.png').exists():lines+=['',f"### {date}",'',f'![Фрагменты {date}]({date}/correspondences.png)']
    lines+=['','## Протокол и ограничения','',
        '0929 — дата разработки. Первый прогон с односторонним порогом 80% сохранён в `development_one_way_080`: геометрия совместилась, но частично перекрывающиеся фрагменты не прошли проверку. Для текущего опыта выбрана двусторонняя проверка частичного перекрытия; строгий исходный критерий оставлен как дополнительная проверка чувствительности. После этого правила и хеши кода зафиксированы в [protocol.json](protocol.json) до прогона шести остальных дат. Пороги по отдельным датам не подстраивались.','',
        'Поиск остаётся эвристикой цвета, плоскостности, формы и высоты над предполагаемой землёй. Фильтр фрагментов мягче, чем в предыдущем опыте: он допускает частичные прямоугольники. Отдельные компоненты могут быть частью земли, навеса или другой конструкции. Семантические классы не обучены. Динамические объекты автоматически не исключены из всей подгонки; оценка без техники дана отдельно по проверенным неподвижным областям.','',
        'Все даты относятся к одной площадке; даже успешный перенос не доказывает работу на новых объектах. RANSAC Open3D может дать отличающиеся фрагменты при повторном запуске. Сохранены фактические точки фрагментов, их описания, начальные матрицы, выбранные пары, результаты и кодовые хеши. Текущая рабочая привязка 3DGS/LAS/BIM не заменялась; эти эксперименты не подтверждают положение самого BIM или его актуальную ревизию.','',
        '## Файлы и воспроизведение','',
        '[Все расстояния](comparison.csv) · [неподвижные конструкции](static_structures.csv) · [все предложенные пары](correspondences.csv) · [проверки сохранённых расчётов](validation.json).','',
        '```sh','OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/align_object_consensus.py 0929 0331 0504 0609 0708 0723 0831',
        '.venv/bin/python scripts/report_object_consensus.py',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/verify_object_consensus.py',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s scripts -p "test_*.py" -v','```','',
        'Шесть новых тестов: известное преобразование с ложными кандидатами; запрет повторного использования одной цели; отказ от тесных/коллинеарных опор; неверные высота, размеры и нормаль; восстановление преобразования по созвездию фрагментов; сохранение ошибочной идентичности пары и двусторонняя оценка перекрытия. Всего 29 тестов.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    (OUT/'static_protocol.json').write_text(json.dumps({'regions_path':str(regions_path.relative_to(ROOT)),
        'regions_sha256':file_hash(regions_path),'original_record_offset':5,'original_record_stride':10,
        'scopes':['unseen_records','spatial_holdout'],'minimum_count_for_adequate_support':30,
        'spatial_tile_width_source_units':.4,'tile_modulo':4,'withheld_remainder':0,
        'selection_frozen_by_existing_baseline':True},indent=2)+'\n')
    print(OUT/'REPORT.md')


if __name__=='__main__':main()
