"""Report a controlled source-label substitution without hiding rejected poses."""
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from align_dates import ROOT,DATES
from align_0929 import apply
from align_object_recovery import load_fragments
from align_object_consensus import patch_checks
from object_consensus import match,descriptors
from diagnose_strict_support import triangle_diagnostics
from check_las_vertical import write_csv,file_hash
from align_semantic_3dgs import OUT,METHOD,PREVIOUS

METHODS=['current_automatic','previous_semantic',METHOD]
NAMES=['Текущий метод','Прежняя семантика','Новые метки 3DGS']


def main():
    audits={d:json.loads((OUT/d/'audit.json').read_text()) for d in DATES};rows=[];statics=[];changes=[];diagnostics={}
    for date,audit in audits.items():
        for method in METHODS:
            row=next((r for r in audit['comparison'] if r['method']==method and r['scope']=='fixed_baseline_coverage'),None)
            rows.append({'date':date,'method':method,'status':audit['comparison_status'][method],
                'strict_pass':audit['comparison_status'][method]=='candidate_for_review' and audit['comparison_strict_support'][method]>=3,
                'strict_support_count':audit['comparison_strict_support'][method],
                **({k:row[k] for k in ['count','median','p90','p95','rmse','within_1m']} if row else {k:None for k in ['count','median','p90','p95','rmse','within_1m']})})
        with (OUT/date/'static_structures.csv').open() as f:statics.extend(csv.DictReader(f))
        run=audit['run'];delta={'date':date,'previous_starts':audit['previous_initial_starts'],'new_starts':audit['initial_starts'],
            'previous_selected':audit['previous_selected_candidate'],'new_selected':run.get('selected_candidate_id'),
            'selected_seed_changed':audit['previous_selected_candidate']!=run.get('selected_candidate_id'),
            'shared_classes':audit['shared_classes'],'source_built_count':audit['source_class_counts']['built'],'target_built_count':audit['target_class_counts']['built']}
        if 'matrix_local' in run:
            held=np.load(OUT/date/'held_out_distances.npz');p=held['source_points'][held['fixed_coverage']];new=apply(p,np.array(run['matrix_local']))
            old=json.loads((PREVIOUS/date/'audit.json').read_text());previous=next(r for r in old['runs'] if r['method']=='semantic_fusion')
            current=json.loads((ROOT/f'results/object_recovery/{date}/selection.json').read_text())
            for name,record in [('current',current),('previous',previous)]:
                distance=np.linalg.norm(new-apply(p,np.array(record['matrix_local'])),axis=1)
                delta[name+'_pose_displacement_median_m']=float(np.median(distance));delta[name+'_pose_displacement_p95_m']=float(np.quantile(distance,.95))
            if date in ['0708','0723']:
                matrix=np.array(run['matrix_local']);sf,tf=load_fragments(date);sd,td=descriptors(sf),descriptors(tf)
                pairs=patch_checks(match(sd,td,matrix),sf,tf,matrix);strict=[p for p in pairs if p['fraction_within_1m']>=.8 and p['reverse_fraction_within_1m']>=.5]
                triangles=triangle_diagnostics(strict,sd,td,matrix)
                previous_ids={(p['source_id'],p['target_id']) for p in previous['final_pair_checks']};new_ids={(p['source_id'],p['target_id']) for p in run['final_pair_checks']}
                diagnostics[date]={'new_bank_strict_pairs':run['strict_overlap_pairs'],'new_bank_triangles':run['strict_triangles'],
                    'new_bank_admissible_triangles':run['admissible_strict_triangles'],
                    'postfit_all_fragments_diagnostic_only':True,'postfit_strict_pairs':strict,'postfit_triangles':triangles,
                    'postfit_admissible_triangles':sum(t['passes'] for t in triangles),
                    'bank_identities_added':[list(p) for p in sorted(new_ids-previous_ids)],'bank_identities_removed':[list(p) for p in sorted(previous_ids-new_ids)]}
        changes.append(delta)
    write_csv(OUT/'summary.csv',rows);write_csv(OUT/'static_structures.csv',statics);write_csv(OUT/'initialization_changes.csv',changes)
    (OUT/'strict_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    find=lambda d,m:next(r for r in rows if r['date']==d and r['method']==m)
    accepted={m:[r['date'] for r in rows if r['method']==m and r['status']=='candidate_for_review'] for m in METHODS}
    strict={m:[r['date'] for r in rows if r['method']==m and r['strict_pass']] for m in METHODS}
    paired=[d for d in accepted[METHOD] if d in accepted['current_automatic']]
    improved=[d for d in paired if find(d,METHOD)['median']<find(d,'current_automatic')['median']-.01]
    worse=[d for d in paired if find(d,METHOD)['median']>find(d,'current_automatic')['median']+.01]
    summary={'accepted_dates':accepted,'strict_dates':strict,'accepted_median_improvements_over_1cm':improved,
        'accepted_median_regressions_over_1cm':worse,'rejected_dates':[d for d in DATES if d not in accepted[METHOD]],
        'changes':changes,'source_sha256':{n:file_hash(ROOT/'scripts'/n) for n in ['report_semantic_3dgs_registration.py']},
        'audit_sha256':{d:file_hash(OUT/d/'audit.json') for d in DATES},'strict_diagnostic_sha256':file_hash(OUT/'strict_diagnostics.json')}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),constrained_layout=True);x=np.arange(7)
    for ax,key,title in zip(axes,['median','p90'],['Медиана расстояния до LAS','90-й процентиль расстояния до LAS']):
        for shift,method,label,color in zip([-.25,0,.25],METHODS,NAMES,['#6b7d92','#8a77ab','#21876f']):
            values=[find(d,method)[key] for d in DATES];ax.bar(x+shift,[v if v is not None else np.nan for v in values],.23,label=label,color=color)
            for i,d in enumerate(DATES):
                if d not in accepted[method] and values[i] is not None:ax.plot(i+shift,values[i],'x',color='#b32937',markersize=8,markeredgewidth=2)
        ax.set(xticks=x,xticklabels=DATES,ylabel='Метры; меньше — лучше',title=title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(fontsize=8);fig.suptitle('Одинаковые отложенные точки и область покрытия · × — отклонённый результат')
    fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
    fmt=lambda v:'—' if v is None else f'{v:.3f}'
    gate=lambda d,m:'да' if d in accepted[m] else 'отказ'
    strict_text=lambda d,m:'да' if d in strict[m] else 'нет'
    table=['| Дата | Текущий: медиана / P90, м | Прежняя семантика: медиана / P90, м | Новые метки: медиана / P90, м | Новый допуск | Строгий: текущий / прежний / новый |','|---|---:|---:|---:|---|---|']
    for d in DATES:table.append('| '+d+' | '+' | '.join(fmt(find(d,m)['median'])+' / '+fmt(find(d,m)['p90']) for m in METHODS)+' | '+gate(d,METHOD)+' | '+' / '.join(strict_text(d,m) for m in METHODS)+' |')
    counts='; '.join(f'{name}: основной {len(accepted[m])}/7, строгий {len(strict[m])}/7' for m,name in zip(METHODS,NAMES))
    conclusions=[]
    for d in ['0708','0723']:
        r=audits[d]['run'];e=diagnostics.get(d)
        if d in strict[METHOD]:text=f'{d}: строгая проверка пройдена, пространственных опор {r["strict_forward_080_spatial_support_count"]}.'
        elif e:
            text=f'{d}: строгая проверка не пройдена. Пар с прямым перекрытием ≥80% и обратным ≥50%: {len(r["strict_overlap_pairs"])}; допустимых троек в исходном банке: {r["admissible_strict_triangles"]}.'
            if r['strict_triangles']:
                t=max(r['strict_triangles'],key=lambda t:min(t['source_second_axis_m'],t['target_second_axis_m']))
                text+=f' Для лучшей тройки вторые пространственные оси 3DGS / LAS равны {t["source_second_axis_m"]:.6f} / {t["target_second_axis_m"]:.6f} м при минимуме 2 м.'
            text+=f' Диагностическое сопоставление всех фрагментов в новой окончательной позе даёт {len(e["postfit_strict_pairs"])} подходящих пар и {e["postfit_admissible_triangles"]} допустимых троек; в допуск оно не вмешивается.'
        else:text=f'{d}: не найден допустимый кандидат для окончательной подгонки.'
        conclusions.append(text)
    lines=['# Совмещение с новыми метками 3DGS: семь дат','',counts+'.','',
        f'Среди принятых результатов улучшений медианы более 1 см относительно текущего метода: {len(improved)}; ухудшений более 1 см: {len(worse)}. Отклонённые результаты: '+(', '.join(summary['rejected_dates']) or 'нет')+'.','',
        '## Сравнение','',*table,'','![Ошибки совмещения](comparison.png)','',
        'Отклонённые результаты показаны с их фактическими ошибками и не считаются успешными. Расстояния — от одинаковых пространственно отложенных центров 3DGS до предоставленного LAS, при неизменной области покрытия. Это внутренняя согласованность, не независимая геодезическая точность. Порог 1 см описывает величину различия, не статистическую значимость.','',
        'Текущий метод — прежний автоматический комбинированный выбор. Прежняя семантика — опыт со специалистом LoveDA и банком пар соседних стартов. Третий вариант заменяет только исходные метки 3DGS на прогнозы по гауссиановому изображению; LAS-метки прежние. Исторические рабочие экспорты просмотрщика отличаются от экспериментального автоматического метода и не заменены.','',
        '## Условия повторного опыта','',
        'Для каждой даты заново оценён весь прежний пул гипотез, выбраны четыре разнесённых старта, выполнен ICP и окончательная подгонка. Классы, их веса, случайное зерно, пороги и бюджет сохранены. Пары соседних гипотез зафиксированы до ICP. Прежние окончательные матрицы не служат стартами и вводятся только после нового расчёта для сравнения.','',
        'В этой реализации семантика влияет на ранжирование стартов и выбор кандидата после грубой подгонки. Окончательная подгонка остаётся геометрической, а фрагменты для строгой проверки не выделяются заново по новым меткам. Поэтому лучшее распознавание класса зданий ещё не исправляет границы или пространственное расположение проверяемых опор.','',
        'Основной допуск требует прежних ограничений масштаба, обучающей медианы ≤0,5 м, цветовой разницы ≤0,35 и трёх разнесённых пар с двусторонним перекрытием ≥50% в пределах 1 м. Строгий допуск дополнительно требует прямого перекрытия ≥80%; расстояние между опорами ≥10 м, вторая сингулярная величина их центрированных координат ≥2 м в обеих сценах. Ни один порог не менялся после просмотра результата.','',
        'Новые метки уже разработаны на этой площадке: это контролируемое сравнение вариантов, не независимый тест переноса. Контрольные области крыши и берега не выбирают позу или параметры.','',
        '## 0708 и 0723','',*sum(([t,''] for t in conclusions),[]),
        'На 0708 в 3DGS много новых меток зданий, но в LAS их только 23 при минимуме 128 для общего класса. Поэтому класс зданий там не участвует в семантическом расстоянии. На 0723 общий класс конструкций появился; его наличие само по себе не гарантирует три надёжные пространственные опоры.','',
        'Подробности всех пар, их прямого/обратного перекрытия и троек сохранены в `<дата>/audit.json` и `strict_diagnostics.json`. Итоговое сопоставление всех фрагментов используется только для объяснения и не добавляет пары задним числом.','',
        '## Изменение инициализации','',
        '| Дата | Прежний выбранный старт | Новый выбранный старт | Общий класс конструкций | Смещение позы к текущей, P95, м |','|---|---|---|---|---:|']
    for c in changes:lines.append(f'| {c["date"]} | {c["previous_selected"]} | {c["new_selected"] or "—"} | {"да" if "built" in c["shared_classes"] else "нет"} | {c.get("current_pose_displacement_p95_m",float("nan")):.6g} |')
    lines+=['','Смещение позы — расстояние между положениями одних и тех же отложенных точек после двух преобразований. Оно отделяет изменение положения от изменения набора подтверждающих пар.','',
        '## Неподвижные конструкции','',
        'Крыша и прибрежная конструкция проверены в прежних фиксированных масках без кранов и техники. `static_structures.csv` содержит отдельные неиспользованные записи и пространственно отложенные выборки, медианы, P90 и число точек для всех трёх методов. Выборки меньше 30 точек помечены как недостаточные; пустые метрики не превращены в нули. Соседние обучающие точки существуют для выборки отдельных записей, поэтому она слабее пространственного отложения.','',
        '## Воспроизведение','',
        '```sh','OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python scripts/align_semantic_3dgs.py','.venv/bin/python scripts/report_semantic_3dgs_registration.py',
        ".venv/bin/python -m unittest discover -s scripts -p 'test_*.py' -v > results/semantic_3dgs_registration/tests.log 2>&1",
        'OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python scripts/verify_semantic_3dgs_registration.py','```','',
        'Прежние метки, геометрические пулы и модели уже находятся в проекте. Новых пакетов и загрузок не требуется. `protocol.json` фиксирует условия и хеши кода; `validation.json` — численную перепроверку всех гипотез, допуска и ошибок; `summary.csv` — таблицу результатов, `initialization_changes.csv` — влияние меток на выбор старта. [Предыдущий опыт распознавания](../semantic_3dgs/REPORT.md).','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    page=['<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Новые метки: повторное совмещение семи дат</title>',
        '<style>body{font:16px/1.6 system-ui,sans-serif;color:#17202a;max-width:1300px;margin:24px auto;padding:0 20px}p{max-width:95ch}img{width:100%;height:auto}a{color:#175794}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:8px;text-align:left;border-bottom:1px solid #ccd2d8}.table{overflow:auto}summary{cursor:pointer;font-weight:650;padding:14px 0}details{border-bottom:1px solid #ccd2d8}a:focus-visible,summary:focus-visible{outline:2px solid #175794}h1{font-size:clamp(24px,3vw,36px)}.rejected{color:#a52635;font-weight:650}</style>',
        '<a href="/viewer/">← 3D-просмотрщик</a><h1>Новые метки: повторное совмещение семи дат</h1>',
        '<p>'+counts+'.</p>',
        f'<p>При прежних порогах и четырёх начальных гипотезах на дату новых принятых улучшений медианы более 1 см: <strong>{len(improved)}</strong>; ухудшений: <strong>{len(worse)}</strong>. Отклонённые результаты показаны в таблице и не считаются успешными.</p>',
        '<p><a href="REPORT.md">Методика и выводы</a> · <a href="summary.csv">Все результаты CSV</a> · <a href="static_structures.csv">Неподвижные конструкции</a> · <a href="../semantic_3dgs/inspection.html">Распознавание до / после</a></p>',
        '<div class="table"><table><thead><tr><th>Дата</th><th>Текущий: медиана / P90, м</th><th>Прежняя семантика, м</th><th>Новые метки, м</th><th>Новый допуск</th><th>Строгий: текущий / прежний / новый</th></tr></thead><tbody>']
    for d in DATES:page.append('<tr><td>'+d+'</td>'+''.join('<td>'+fmt(find(d,m)['median'])+' / '+fmt(find(d,m)['p90'])+'</td>' for m in METHODS)+f'<td class="{"rejected" if d not in accepted[METHOD] else ""}">{gate(d,METHOD)}</td><td>'+ ' / '.join(strict_text(d,m) for m in METHODS)+'</td></tr>')
    page+=['</tbody></table></div><p>Одинаковые отложенные точки; меньшее расстояние лучше. Это согласованность с данным LAS. Полные рабочие сцены и BIM сохранены.</p>',
        '<img src="comparison.png" alt="Медиана и P90 расстояний до LAS для трёх вариантов на семи датах">',
        '<h2>Строгая проверка проблемных дат</h2>',*['<p>'+t+'</p>' for t in conclusions],
        '<p>На 0708 общий класс зданий недоступен: в LAS только 23 уверенные точки при минимуме 128. На 0723 этот класс появился. Все пороги сохранены.</p>']
    for d in ['0708','0723']:
        r=audits[d]['run'];strict_ids={(p['source_id'],p['target_id']) for p in r.get('strict_overlap_pairs',[])}
        page.append(f'<details><summary>{d}: проверка каждой исходной пары</summary><div class="table"><table><thead><tr><th>3DGS → LAS</th><th>Прямое перекрытие</th><th>Обратное перекрытие</th><th>Строгое перекрытие и назначение</th></tr></thead><tbody>')
        for p in r.get('final_pair_checks',[]):page.append(f'<tr><td>{p["source_id"]} → {p["target_id"]}</td><td>{p["fraction_within_1m"]*100:.1f}%</td><td>{p["reverse_fraction_within_1m"]*100:.1f}%</td><td>{"да" if (p["source_id"],p["target_id"]) in strict_ids else "нет"}</td></tr>')
        page.append('</tbody></table></div><p>Для окончательного строгого допуска ещё нужны минимум три пространственно разнесённые опоры.</p></details>')
    (OUT/'inspection.html').write_text('\n'.join(page)+'</html>\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
