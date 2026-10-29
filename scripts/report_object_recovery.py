"""Report scale recovery without changing the working 3DGS/LAS/BIM registration."""
import json

import laspy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from align_dates import ROOT,ORIGIN,DATES
from align_0929 import apply,summary
from align_object_recovery import OUT
from check_static_structures import volume_mask
from check_las_vertical import write_csv


def draw(date,run,target,color):
    if 'matrix_local' not in run:return
    folder=ROOT/f'results/object_consensus/{date}';out=OUT/date
    source=np.load(folder/'source_candidate_points.npz');dest=np.load(folder/'target_candidate_points.npz')
    pairs=run['final_spatial_support'] or run['final_pair_checks'][:5]
    if not pairs:return
    moved=[apply(source[p['source_id']],np.array(run['matrix_local'])) for p in pairs]
    refs=[dest[p['target_id']] for p in pairs];joint=np.concatenate(moved+refs)
    lo=joint[:,:2].min(0)-8;hi=joint[:,:2].max(0)+8
    keep=np.all((target[:,:2]>=lo)&(target[:,:2]<=hi),axis=1);background=target[keep];rgb=color[keep]
    order=np.argsort(background[:,2])[::3];fig,axes=plt.subplots(1,2,figsize=(12,6),constrained_layout=True)
    for ax,patches,label in zip(axes,[moved,refs],['Фрагменты 3DGS после совмещения','Предложенные фрагменты LAS']):
        ax.scatter(background[order,0],background[order,1],c=rgb[order],s=1,alpha=.55)
        for i,patch in enumerate(patches):
            c=plt.get_cmap('tab10')(i%10);shown=patch[::max(1,len(patch)//300)]
            ax.scatter(shown[:,0],shown[:,1],s=5,color=c);center=patch.mean(0)
            ax.text(center[0],center[1],str(i+1),bbox=dict(facecolor='white',edgecolor=c,boxstyle='round,pad=.2'))
        ax.set(aspect='equal',xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),title=label,xlabel='X, м',ylabel='Y, м')
    fig.suptitle(f"{date} · {len(run['final_spatial_support'])} разнесённых пар после проверки")
    fig.savefig(out/'correspondences.png',dpi=150);plt.close(fig)


def main():
    audits={d:json.loads((OUT/d/'audit.json').read_text()) for d in DATES}
    selections={d:json.loads((OUT/d/'selection.json').read_text()) for d in DATES}
    old={d:json.loads((ROOT/f'results/object_consensus/{d}/audit.json').read_text()) for d in DATES}
    regions=json.loads((ROOT/'results/static_structures/regions.json').read_text())['regions']
    rows=[];statics=[];pairs=[];lines=[]
    for date,audit in audits.items():
        rows.extend(audit['comparison']);las=laspy.read(ROOT/f'assets/{date}/{date}.las')
        target=np.c_[las.x,las.y,las.z]-ORIGIN;color=np.c_[las.red,las.green,las.blue].astype(float);color/=65535 if color.max()>255 else 255
        controls=np.load(ROOT/f'results/object_consensus/{date}/static_control_source_points.npz')
        matrices={r['method']:np.array(r['matrix_local']) for r in audit['runs'] if 'matrix_local' in r}
        matrices['previous_consensus']=np.array(old[date]['runs'][1]['matrix_local'])
        base=ROOT/('results/0929_baseline' if date=='0929' else f'results/multidate/{date}')
        matrices['working_baseline']=np.array(json.loads((base/'alignment.json').read_text())['ply_to_las_local'])
        if selections[date]['selected_method']:matrices['hybrid']=np.array(selections[date]['matrix_local'])
        selected=selections[date]['selected_method']
        if selected:rows.extend({**r,'method':'hybrid'} for r in audit['comparison'] if r['method']==selected)
        for region in regions:
            tree=cKDTree(target[volume_mask(target,region)])
            for scope in ['unseen_records','spatial_holdout']:
                points=controls[region['id']+'_'+scope]
                for method,matrix in matrices.items():
                    distance=tree.query(apply(points,matrix))[0]
                    statics.append({'date':date,'region':region['id'],'scope':scope,'method':method,
                        **(summary(distance) if len(distance) else {'count':0,'median':None,'p90':None,'p95':None,'rmse':None}),
                        'adequate_support':len(distance)>=30})
        for run in audit['runs']:
            pairs.extend({'date':date,'method':run['method'],**p} for p in run.get('final_pair_checks',[]))
        draw(date,next(r for r in audit['runs'] if r['method']=='pair_preserving'),target,color)
    write_csv(OUT/'comparison.csv',rows);write_csv(OUT/'static_structures.csv',statics);write_csv(OUT/'correspondences.csv',pairs)
    def run(date):return next(r for r in audits[date]['runs'] if r['method']=='pair_preserving')
    def metric(date,method):return next((r for r in rows if r['date']==date and r['method']==method and r['scope']=='fixed_baseline_coverage'),None)
    def value(row,key='median'):return '—' if row is None or row[key] is None else f"{row[key]:.3f}"
    passed=[d for d in DATES if run(d)['status']=='candidate_for_review']
    strict=[d for d in passed if run(d)['strict_forward_080_spatial_support_count']>=3]
    hybrid_passed=[d for d in DATES if selections[d]['selected_method']]
    hybrid_strict=[d for d in hybrid_passed if selections[d]['strict_forward_080_spatial_support_count']>=3]
    lines=['# Исправление начального масштаба и сохранение соответствий','',
        f"**Итог комбинированного выбора: {len(hybrid_passed)} из 7 дат проходят основной фильтр; {len(hybrid_strict)} — дополнительный строгий фильтр.** Прежний автоматический результат сохраняется, если он прошёл проверки; новый способ используется только при его отказе. Выбор не использует дату, контрольную ошибку, RTK или рабочую матрицу. Это ограниченный результат на изученной площадке, не независимая геодезическая приёмка.",'',
        '## Итог по единому правилу выбора','',
        '| Дата | Выбранный автоматический способ | Медиана / P90, м | Разнесённых пар | Строгих пар |',
        '|---|---|---:|---:|---:|']
    for d in DATES:
        r=selections[d];m=metric(d,'hybrid');label='прежний консенсус' if r['selected_method']=='previous_consensus' else 'масштаб по паре' if r['selected_method'] else 'отказ'
        lines.append(f"| {d} | {label} | {value(m)} / {value(m,'p90')} | {len(r.get('final_spatial_support',[]))} | {r.get('strict_forward_080_spatial_support_count',0)} |")
    lines+=['','Правило реализовано в `scripts/select_object_alignment.py`: сначала проверяются обучающая медиана ≤0,5 м, различие RGB ≤0,35 и минимум три подтверждённые пары. При успехе прежнего автоматического метода он сохраняется; иначе проверяется новый. При отказе обоих матрица не выдаётся. Числа ниже также показывают неудачи нового варианта отдельно — улучшение 0708/0723 не оправдывает ухудшение других дат.','',
        f"Отдельно новый способ проходит основной фильтр на **{len(passed)} из 7 дат**: {', '.join(passed) or 'ни одна'}. Дополнительный строгий фильтр перекрытия 80% проходят **{len(strict)}**: {', '.join(strict) or 'ни одна'}. Это геометрически согласованные кандидаты, не независимая геодезическая точность и не распознавание классов объектов.",'',
        '## Причина прежних сбоев','',
        'В прошлой версии начальный масштаб зависел от размеров целых сцен. При разном охвате съёмок этот признак оказался ненадёжным: для 0708 он давал 47,06 против примерно 31,37 после геометрического уточнения; для 0723 — 32,64 против примерно 22,92. Это расхождение около 50% и 42%. Диагностика прежней итоговой матрицей показала, что подходящие фрагменты в сохранённых наборах есть. Эта диагностика не использовалась для построения новых начальных матриц.','',
        'Теперь масштаб оценивается по расстоянию между двумя предполагаемыми соответствующими фрагментами. Направление этой пары и нормали поверхностей задают поворот, центры — сдвиг. Остальные фрагменты проверяют гипотезу. После грубого ICP допускаются только начальные гипотезы, сохранившие не менее трёх пространственно разнесённых соответствий; идентификаторы пар не заменяются ближайшими новыми.','',
        '**Допущение о масштабе:** съёмки относятся к одной площадке и имеют сопоставимый охват. Размер всей сцены теперь задаёт только широкий допустимый интервал: 0,4–1,6 от прежней грубой оценки. Начальные матрицы вне него отбрасываются; выход за интервал после уточнения также означает отказ. Масштаб внутри интервала оценивается по расстояниям между фрагментами. Для очень маленького фрагмента площадки это допущение может не выполняться — универсальная регистрация произвольных частичных сцен не заявляется.','',
        '## Новый вариант отдельно: одинаковые контрольные точки','',
        'Основные числа — расстояния центров гауссианов до ближайших точек LAS в заранее фиксированной области покрытия. Полные ошибки вне этой области также сохранены. Область и контрольные записи совпадают с предыдущим опытом.','',
        '| Дата | Прежний метод: медиана, м | Новый: медиана / P90, м | Рабочая привязка: медиана, м | Разнесённых пар: раньше → сейчас | Строгих пар | Статус |',
        '|---|---:|---:|---:|---:|---:|---|']
    for d in DATES:
        r=run(d);prev=old[d]['runs'][1];m=metric(d,'pair_preserving')
        status='кандидат для проверки' if r['status']=='candidate_for_review' else r['status']
        lines.append(f"| {d} | {value(metric(d,'previous_consensus'))} | {value(m)} / {value(m,'p90')} | {value(metric(d,'working_baseline'))} | {len(prev['final_spatial_support'])} → {len(r.get('final_spatial_support',[]))} | {r.get('strict_forward_080_spatial_support_count',0)} | {status} |")
    lines+=['','Основной фильтр перекрытия прежний: минимум 50% точек ближе 1 м **в обоих направлениях**, согласованные размеры/нормали и центры в пределах 5 м; минимум три пары с расстоянием между опорами ≥10 м и достаточным отклонением от одной прямой. Строгая проверка дополнительно требует ≥80% в направлении 3DGS → LAS. Для старой и новой версии одинаковые пороги.','',
        '## Отдельно проверен вклад отбора после грубого ICP','',
        'Оба варианта используют одинаковые новые начальные гипотезы и до четырёх одинаковых грубых уточнений. Контроль выбирает наименьшую обучающую геометрическую ошибку; новый вариант выбирает среди гипотез с сохранёнными парами. Совпадающий выбор уточняется только один раз.','',
        '| Дата | Индекс выбранной гипотезы: геометрия / сохранение пар | Медиана: геометрия / сохранение пар, м |',
        '|---|---:|---:|']
    for d in DATES:
        control=audits[d]['runs'][0];r=run(d)
        lines.append(f"| {d} | {control.get('selected_seed_index','—')} / {r.get('selected_seed_index','—')} | {value(metric(d,'geometric_control'))} / {value(metric(d,'pair_preserving'))} |")
    same=sum(audits[d]['runs'][0].get('selected_seed_index')==run(d).get('selected_seed_index') for d in DATES)
    lines+=['',f'Выбор совпал на {same} датах. Если оба варианта выбирают одну гипотезу, улучшение нельзя приписывать новому фильтру: в этом случае оно связано с новым набором начальных матриц.','',
        '## Неподвижные конструкции','',
        'Сохранены прежние объёмы крыши и прибрежной конструкции, исключающие краны и дорожную технику из этой оценки. Использованы те же нечётные исходные записи, которые не участвовали в подборе матрицы. Они могут соседствовать с обучающими точками, поэтому это не пространственно независимая проверка. Более строгие отложенные плитки также сохранены; в обоих контрольных объёмах одновременно нет достаточных ≥30 точек ни на одной дате. Это ограничение остаётся.','',
        '| Дата | Крыша: комбинированный / рабочий, м | Берег: комбинированный / рабочий, м | Точек: крыша / берег |',
        '|---|---:|---:|---:|']
    for d in DATES:
        def s(region,method):return next((r for r in statics if r['date']==d and r['region']==region and r['scope']=='unseen_records' and r['method']==method),None)
        a,b=s('roof','working_baseline'),s('bank_mid','working_baseline')
        lines.append(f"| {d} | {value(s('roof','hybrid'))} / {value(a)} | {value(s('bank_mid','hybrid'))} / {value(b)} | {a['count']} / {b['count']} |")
    for d in ['0708','0723']:
        if (OUT/d/'correspondences.png').exists():lines+=['',f'## Соответствия {d}','',f'![Фрагменты {d}]({d}/correspondences.png)']
    lines+=['','## Воспроизводимость и пределы вывода','',
        '0708 и 0723 стали датами разработки исправления. Остальные пять — проверка отсутствия ухудшения на уже изученной площадке, не слепой внешний тест. Настройки зафиксированы в [protocol.json](protocol.json). Использован прежний сохранённый набор фрагментов без нового RANSAC: изменение относится к построению и отбору матриц, а не к случайно другой сегментации.','',
        'Правило комбинирования добавлено после обнаружения ухудшений 0504 и 0929; оно считается частью разработки на этих данных. Оно не содержит исключений по датам и не выбирает метод по отложенным метрикам. Дополнительного доказательства обобщения на новую площадку это не даёт.','',
        'Неограниченный промежуточный опыт сохранён в `../object_recovery_unbounded`, вместе со снимком исходного кода. На 0708 он восстановил пары, но на 0723 и 0504 сжал сцену до недопустимого масштаба; пакет был остановлен во время расчёта 0609. После этого введён общий диапазон масштаба и повторены все семь дат. Сохранённые независимые начальные матрицы переиспользуются там, где доступны, с хешем файла и индексом выбранной матрицы. Неудачные промежуточные результаты не заменяют итоговые строки.','',
        'Новые матрицы не используют RTK/COLMAP, размеченную крышу, ручной старт 0708 или прежнюю матрицу. Старые матрицы участвуют только в сравнении после выбора новых результатов. Подбор использует прежние обучающие плитки; контрольные точки не участвуют в ранжировании или подгонке.','',
        'Поверхности всё ещё выделяются по геометрии и цвету. Совпавший фрагмент не означает распознанный класс здания; некоторые фрагменты могут принадлежать временным объектам. Отсутствие техники проверяется только внутри явно выбранных неподвижных контрольных объёмов. Рабочие экспорты и положение BIM не заменены; независимая проверка системы координат и актуальной ревизии BIM остаётся отдельной задачей.','',
        '## Файлы','',
        '[Выбранные матрицы](selection_summary.json) · [все расстояния](comparison.csv) · [неподвижные конструкции](static_structures.csv) · [пары](correspondences.csv) · [численные проверки](validation.json). Итог комбинированного выбора для каждой даты находится в `selection.json`; `audit.json` содержит отдельные испытания, включая отклонённые. Матрицы переводят исходные центры 3DGS в локальные координаты LAS с прежним началом [825800, 842100, 0]. Это не экспорт полных гауссианов: для него также требуется преобразование ориентаций, масштабов и сферических гармоник.','',
        '```sh','OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/align_object_recovery.py 0708 0723 0331 0504 0609 0831 0929',
        '.venv/bin/python scripts/select_object_alignment.py',
        '.venv/bin/python scripts/report_object_recovery.py',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/verify_object_recovery.py','```','',
        'Добавлено десять тестов: шесть для новой инициализации, вырожденных случаев, масштаба и сохранения идентичности; четыре для выбора метода, сохранения принятого результата, отказа обоих методов и перепроверки критериев. Всего 39 тестов.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines));print(OUT/'REPORT.md')


if __name__=='__main__':main()
