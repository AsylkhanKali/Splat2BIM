"""Publish both pre-ICP variants, including rejected fits, for every date."""
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from align_dates import ROOT,DATES
from check_las_vertical import write_csv
from align_semantic_initial import OUT

METHODS=['current_automatic','pooled_geometry','semantic_initial']
NAMES=['Текущий комбинированный','Геометрический выбор до ICP','Семантический выбор до ICP']


def main():
    audits={d:json.loads((OUT/d/'audit.json').read_text()) for d in DATES}
    rows=[r for d in DATES for r in audits[d]['comparison']];write_csv(OUT/'comparison.csv',rows)
    def metric(date,method):return next((r for r in rows if r['date']==date and r['method']==method and r['scope']=='fixed_baseline_coverage'),None)
    def run(date,method):return next(r for r in audits[date]['runs'] if r['method']==method)
    def status(date,method):return audits[date]['current_status'] if method=='current_automatic' else run(date,method)['status']
    def value(row,key='median'):return '—' if row is None or row.get(key) is None else f"{row[key]:.3f}"
    accepted={m:[d for d in DATES if status(d,m)=='candidate_for_review'] for m in METHODS}
    strict={m:[d for d in accepted[m] if (audits[d]['current_strict_support'] if m==METHODS[0] else run(d,m).get('strict_forward_080_spatial_support_count',0))>=3] for m in METHODS}
    recovered=[d for d in accepted[METHODS[2]] if d not in accepted[METHODS[1]]]
    lost=[d for d in accepted[METHODS[1]] if d not in accepted[METHODS[2]]]
    table=[]
    for d in DATES:
        old,geo,sem=[metric(d,m) for m in METHODS]
        table.append({'date':d,'initial_hypotheses':audits[d]['initial_pool_size'],
            'current_median_m':old['median'],'current_p90_m':old['p90'],
            'geometry_median_m':geo['median'] if geo else None,'geometry_p90_m':geo['p90'] if geo else None,
            'geometry_status':status(d,METHODS[1]),'semantic_median_m':sem['median'] if sem else None,
            'semantic_p90_m':sem['p90'] if sem else None,'semantic_status':status(d,METHODS[2]),
            'semantic_minus_current_m':sem['median']-old['median'] if sem else None,
            'semantic_minus_geometry_m':sem['median']-geo['median'] if sem and geo else None,
            'geometry_start':run(d,METHODS[1]).get('selected_candidate_id'),
            'semantic_start':run(d,METHODS[2]).get('selected_candidate_id')})
    write_csv(OUT/'summary.csv',table)
    statics=[]
    for d in DATES:
        with (OUT/d/'static_structures.csv').open() as f:statics.extend(csv.DictReader(f))
    write_csv(OUT/'static_structures.csv',statics)
    static_summary=[]
    for d in DATES:
        for region in ['roof','bank_mid']:
            r=[row for row in statics if row['date']==d and row['region']==region and row['scope']=='unseen_records']
            old=next(x for x in r if x['method']=='current_automatic');sem=next((x for x in r if x['method']=='semantic_initial'),None)
            static_summary.append({'date':d,'region':region,'count':int(old['count']),
                'current_median_m':float(old['median']) if old['median'] else None,
                'semantic_median_m':float(sem['median']) if sem and sem['median'] else None,
                'current_p90_m':float(old['p90']) if old['p90'] else None,
                'semantic_p90_m':float(sem['p90']) if sem and sem['p90'] else None,
                'semantic_status':status(d,'semantic_initial')})
    write_csv(OUT/'static_summary.csv',static_summary)
    deltas=[r['semantic_minus_current_m'] for r in table if r['date'] in accepted['semantic_initial']]
    summary={'accepted_dates':accepted,'strict_support_dates':strict,'semantic_recovers_geometric_failures':recovered,
        'semantic_loses_geometric_successes':lost,'semantic_accepted_improved_median_over_1cm':sum(v<-.01 for v in deltas),
        'semantic_accepted_worse_median_over_1cm':sum(v>.01 for v in deltas),
        'semantic_accepted_within_1cm':sum(abs(v)<=.01 for v in deltas)}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(13,4.6),constrained_layout=True);x=np.arange(7);width=.25
    for i,(method,name,color) in enumerate(zip(METHODS,NAMES,['#5d79aa','#bd7538','#269273'])):
        for ax,key in zip(axes,['median','p90']):
            values=[metric(d,method)[key] if metric(d,method) else np.nan for d in DATES]
            bars=ax.bar(x+(i-1)*width,values,width,color=color,label=name)
            for bar,date in zip(bars,DATES):
                if date not in accepted[method]:bar.set_hatch('///');bar.set_alpha(.55)
            ax.set(xticks=x,xticklabels=DATES,ylabel='м',title='Медиана' if key=='median' else '90-й процентиль')
            ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(fontsize=8);fig.suptitle('Одинаковые отложенные точки · штриховка означает отклонённый результат')
    fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
    accepted_text='; '.join(f'{name}: {len(accepted[m])}/7' for m,name in zip(METHODS,['текущий','геометрический контроль','семантика']))
    conclusion=f"Основной фильтр совмещения: **{accepted_text}**. Семантика восстановила отказ геометрического контроля на датах: {', '.join(recovered) or 'нет'}. Потеряла удачный геометрический результат на датах: {', '.join(lost) or 'нет'}."
    recommendation='Текущий комбинированный метод остаётся рекомендуемым: семантика помогает выбрать старт в отдельных случаях, но не улучшает конечную точность и теряет подтверждение соответствий на 0723.'
    lines=['# Семантический выбор начального совмещения','',conclusion,'',recommendation,'',
        f"Среди принятых семантических результатов относительно текущего комбинированного метода: улучшений медианы более 1 см — **{summary['semantic_accepted_improved_median_over_1cm']}**, ухудшений — **{summary['semantic_accepted_worse_median_over_1cm']}**, различий в пределах 1 см — **{summary['semantic_accepted_within_1cm']}**. Порог 1 см описывает величину изменения; статистической значимости он не устанавливает.",'',
        '## Сравнение всех семи дат','',
        '| Дата | Гипотез до ICP | Текущий, медиана / P90, м | Геометрический контроль, медиана / P90, м | Семантика, медиана / P90, м | Допуск: геометрия / семантика |',
        '|---|---:|---:|---:|---:|---|']
    for r in table:
        d=r['date'];old,geo,sem=[metric(d,m) for m in METHODS]
        gates=' / '.join('да' if d in accepted[m] else 'отказ' for m in METHODS[1:])
        lines.append(f"| {d} | {r['initial_hypotheses']} | {value(old)} / {value(old,'p90')} | {value(geo)} / {value(geo,'p90')} | {value(sem)} / {value(sem,'p90')} | {gates} |")
    lines+=['','![Сравнение трёх методов](comparison.png)','',
        'Метрики отклонённых матриц показаны для прозрачности, но эти матрицы не считаются успешным совмещением. Низкая ближайшая дистанция сама по себе не защищает от сжатия сцены и неверных соответствий. Допуск требует допустимого масштаба, обучающей медианы ≤0,5 м, различия RGB ≤0,35 и ≥3 пространственно разнесённых пар с двусторонним подтверждением.','',
        'Строгая проверка минимум трёх пар с дополнительной долей исходных точек ближе 1 м ≥80%: '+', '.join(f'{name} {len(strict[m])}/7' for m,name in zip(METHODS,['текущий метод','геометрический контроль','семантика']))+'. Количество подтверждённых пар характеризует набор соответствий; его рост при той же матрице не означает улучшения точности положения.','',
        '## Как устроен опыт','',
        'Текущий метод в этом сравнении — автоматическое правило выбора из `object_recovery/selection.json`, без RTK и ручного старта. Полные сцены просмотрщика используют ранее проверенные рабочие экспорты; это отдельный baseline. Все расстояния ниже относятся к согласованности с данным LAS, а не к независимой геодезической точности. Контроль использует прежние пространственно отложенные точки и одинаковую фиксированную область покрытия LAS.','',
        'Использована предобученная [SegFormer-B0 для изображений с дрона](https://huggingface.co/Thalirajesh/Aerial-Drone-Image-Segmentation). Сеть получает отдельные цветные проекции обучающих центров 3DGS и LAS. Метки конструкций, покрытий, грунта и растительности образуют дополнительные 3D-соответствия: исходная точка ищет ближайшую точку того же класса. Неуверенные метки, вода и подвижные классы в выборе не участвуют. Сеть не дообучается на этих семи датах.','',
        'Для каждой даты объединены все сохранённые исходные гипотезы геометрического консенсуса и восстановления масштаба, до ICP. Убраны недопустимые масштабы и дубликаты матриц с точностью округления 1e-8. Оба новых варианта используют один набор и одинаковый бюджет: до четырёх стартов, разнесённых минимум на 2 м по контрольным якорям. Старты оцениваются только на обучающих данных.','',
        'Геометрический балл до ICP — P75 расстояния 512 одинаково выбранных обучающих проб до LAS. Семантический вариант добавляет половину среднего по классам P60 расстояния до точек того же класса, ограниченного 20 м. Каждый общий класс имеет равный вес; нужны минимум два класса по ≥128 точек, используется до 512 исходных точек каждого класса. Параметры совпадают с первым опытом, качество по отложенным точкам их не меняет.','',
        'После выбора стартов выполняются одинаковые 40 итераций на каждом из четырёх порогов ICP. Геометрические условия допуска coarse-кандидатов одинаковы. Выбор после coarse-этапа также использует соответствующий балл; финальная подгонка point-to-plane одинакова. Текущие окончательные матрицы, RTK, ручной старт 0708, контрольная крыша и отложенные ошибки не выбирают новые матрицы.','',
        'Первый [опыт выбора после coarse-ICP](../semantic_matching/REPORT.md) сохранён полностью: там семантика поменяла выбор на трёх датах, но финальные матрицы практически совпали. Настоящий опыт переносит вмешательство до ICP, когда выбор старта ещё может менять результат. Это дополнительный исследовательский опыт на известной площадке, а не независимое подтверждение обобщения.','',
        '## Неподвижные конструкции','',
        '| Дата | Участок | Контрольных точек | Текущий: медиана / P90, м | Семантика: медиана / P90, м |',
        '|---|---|---:|---:|---:|']
    for r in static_summary:
        lines.append(f"| {r['date']} | {'крыша' if r['region']=='roof' else 'берег'} | {r['count']} | {value(r,'current_median_m')} / {value(r,'current_p90_m')} | {value(r,'semantic_median_m')} / {value(r,'semantic_p90_m')} |")
    lines+=['','Использованы те же маски вне кранов и техники и те же отдельные записи, что в предыдущем контроле. Эти записи непосредственно не подгонялись, но пространственно соседствуют с обучением. Строго отложенные пространственные точки сохранены отдельно в `static_structures.csv`; на узких конструкциях их мало. Пустая выборка не заменяется нулём. Метрики отклонённой семантической матрицы, если она рассчитана, остаются диагностическими.','',
        '## Что ограничивает результат','',
        'На цветных проекциях центров гауссианов модель выдаёт очень мало уверенных меток конструкций. Главную роль играют грунт и растительность. На всех семи заранее выбранных крышах LAS уверенные метки класса «конструкции» отсутствуют; это конкретный неудачный случай модели, а не подтверждение качества сегментации. Отдельная диагностика обычного фото 0929 показывает метки конструкций, но не даёт размеченного теста точности.','',
        'Изображения LAS и центров гауссианов отличаются текстурой, шумом и пропусками. Семантические прогнозы надо улучшать и проверять на размеченной выборке, прежде чем заменять ими проверенный геометрический метод. Эти эксперименты не доказывают, что семантика повышает абсолютную точность. BIM остаётся в принятой общей системе координат и не используется как подгоняемая цель.','',
        '[Все карты классов и покрытие](../semantic_matching/inspection.html) · [CSV классов](../semantic_matching/semantic_coverage.csv) · [исходное фото для диагностики](../semantic_matching/photo_diagnostic/0929_photo_rgb.png) · [его прогноз](../semantic_matching/photo_diagnostic/0929_photo_semantic.png).','',
        '## Файлы и повторение','',
        '`summary.csv` содержит сводку, `comparison.csv` — обе области контроля, `static_structures.csv` — обе выборки неподвижных конструкций. В каждой папке даты сохранены все баллы исходных гипотез, выбранные старты, матрицы, исходные идентичности пар, статусы и ошибки. `protocol.json` фиксирует параметры и хеши кода до запуска.','',
        '```sh','OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python scripts/align_semantic_initial.py',
        '.venv/bin/python scripts/report_semantic_initial.py',
        "OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -m unittest discover -s scripts -p 'test_*.py' -v > results/semantic_initial/tests.log 2>&1",
        'OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python scripts/verify_semantic_initial.py','```','',
        'Сначала должны быть подготовлены метки командой `prepare_semantic_clouds.py`; веса и версии библиотек зафиксированы в отчёте первого опыта. Проверки второго опыта сохранены в `validation.json`. Рабочие полные PLY и положение BIM не заменены экспериментальными матрицами.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    page=['<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Семантика до ICP: семь дат</title>',
        '<style>body{font:16px/1.6 system-ui,sans-serif;color:#17202a;background:white;max-width:1300px;margin:24px auto;padding:0 20px}p{max-width:80ch}img{width:100%;height:auto}a{color:#175794}a:focus-visible,summary:focus-visible{outline:2px solid #175794}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:8px;text-align:left;border-bottom:1px solid #ccd2d8}summary{cursor:pointer;font-weight:650;padding:14px 0}details{border-bottom:1px solid #ccd2d8}.table{overflow:auto}h1{font-size:clamp(24px,3vw,36px)}</style>',
        '<a href="/viewer/">← 3D-просмотрщик</a><h1>Семантический выбор до ICP</h1>',
        f'<p>Основной фильтр: {accepted_text}. Семантика восстановила отказ геометрического контроля на датах: {", ".join(recovered) or "нет"}. Потеряла удачный результат на датах: {", ".join(lost) or "нет"}.</p>',
        f'<p><strong>{recommendation}</strong></p>',
        '<p>Оба новых варианта выбирают четыре старта из одинакового набора гипотез. Семантика добавляет соответствия между точками одного предсказанного класса. Финальная геометрическая подгонка одинакова.</p>',
        '<p><a href="REPORT.md">Полный отчёт</a> · <a href="summary.csv">Сводка CSV</a> · <a href="static_structures.csv">Неподвижные конструкции</a> · <a href="../semantic_matching/inspection.html">Первый опыт и карты классов</a></p>',
        '<div class="table"><table><thead><tr><th>Дата</th><th>Текущий, м</th><th>Геометрия, м</th><th>Семантика, м</th></tr></thead><tbody>']
    for d in DATES:
        cells=''.join(f'<td>{value(metric(d,m))}{" · отказ" if d not in accepted[m] else ""}</td>' for m in METHODS)
        page.append(f'<tr><td>{d}</td>{cells}</tr>')
    page+=['</tbody></table></div><p>Медиана до LAS на одинаковых отложенных точках. Отклонённая матрица не считается успешным совмещением.</p>',
        '<img src="comparison.png" alt="Медиана и P90 для трёх методов; штриховкой отмечены отклонённые результаты">',
        '<p>На проекциях 3DGS уверенных меток зданий почти нет. Сеть в основном использует грунт и растительность. Это ограниченный эксперимент на одной площадке.</p>']
    for d in DATES:page.append(f'<details><summary>{d}: цвет и семантические классы</summary><a href="../semantic_matching/{d}/semantics.png"><img loading="lazy" src="../semantic_matching/{d}/semantics.png" alt="{d}: отдельно построенные проекции 3DGS и LAS и прогнозы классов"></a></details>')
    (OUT/'inspection.html').write_text('\n'.join(page)+'</html>\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
