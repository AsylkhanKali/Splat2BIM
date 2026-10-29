"""Report the new labels, correspondence-bank ablation and all seven dates."""
import csv
import json

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from align_dates import ROOT,DATES
from check_las_vertical import write_csv
from semantic_projection import COLORS
from align_semantic_fusion import OUT

METHODS=['current_automatic','previous_semantic','geometry_bank','old_semantic_bank','semantic_fusion']
NAMES=['Текущий комбинированный','Прежняя семантика','Геометрия + банк пар','Прежние метки + банк пар','Новые метки + банк пар']


def main():
    audits={d:json.loads((OUT/d/'audit.json').read_text()) for d in DATES}
    previous={d:json.loads((ROOT/f'results/semantic_initial/{d}/audit.json').read_text()) for d in DATES}
    def run(d,m):return next(r for r in (previous[d] if m=='previous_semantic' else audits[d])['runs'] if r['method']==('semantic_initial' if m=='previous_semantic' else m))
    def status(d,m):return audits[d]['current_status'] if m=='current_automatic' else run(d,m)['status']
    def strict(d,m):return audits[d]['current_strict_support'] if m=='current_automatic' else run(d,m).get('strict_forward_080_spatial_support_count',0)
    def metric(d,m):return next((r for r in (previous[d] if m=='previous_semantic' else audits[d])['comparison'] if r['method']==('semantic_initial' if m=='previous_semantic' else m) and r['scope']=='fixed_baseline_coverage'),None)
    def value(d,m,k='median'):
        r=metric(d,m);return '—' if r is None else f"{r[k]:.3f}"
    accepted={m:[d for d in DATES if status(d,m)=='candidate_for_review'] for m in METHODS}
    summary={'accepted_dates':accepted,'strict_dates':{m:[d for d in accepted[m] if strict(d,m)>=3] for m in METHODS}}
    rows=[]
    for d in DATES:
        for m in METHODS:
            r=metric(d,m)
            rows.append({'date':d,'method':m,'status':status(d,m),'median_m':r['median'] if r else None,'p90_m':r['p90'] if r else None,'strict_pairs':strict(d,m)})
    write_csv(OUT/'summary.csv',rows)
    summary['new_accepted_median_improvements_over_1cm']=sum(metric(d,'semantic_fusion')['median']<metric(d,'current_automatic')['median']-.01 for d in accepted['semantic_fusion'])
    summary['new_accepted_median_regressions_over_1cm']=sum(metric(d,'semantic_fusion')['median']>metric(d,'current_automatic')['median']+.01 for d in accepted['semantic_fusion'])
    with (OUT/'recognition.csv').open() as f:recognition=list(csv.DictReader(f))
    def rec(d,side,region,method):return next(r for r in recognition if r['date']==d and r['side']==side and r['region']==region and r['method']==method)
    roof=[float(rec(d,'target','roof','fusion')['built_fraction']) for d in DATES]
    summary['las_roof_detected_dates']=[d for d,p in zip(DATES,roof) if p>0]
    summary['source_roof_detected_dates']=[d for d in DATES if float(rec(d,'source','roof','fusion')['built_fraction'])>0]
    summary['las_vegetation_false_built_max']=max(float(rec(d,'target','vegetation','fusion')['built_fraction']) for d in DATES)
    summary['shared_built_class_dates']=[d for d in DATES if min(int(rec(d,s,'all_visible','fusion')['built_points']) for s in ['source','target'])>=128]
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    statics=[]
    for d in DATES:
        with (OUT/d/'static_structures.csv').open() as f:statics.extend(csv.DictReader(f))
    write_csv(OUT/'static_structures.csv',statics)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True);x=np.arange(7)
    axes[0].bar(x-.17,[float(rec(d,'target','roof','previous')['built_fraction'])*100 for d in DATES],.34,label='Прежняя модель',color='#697b92')
    axes[0].bar(x+.17,np.array(roof)*100,.34,label='Новая модель зданий',color='#2c8a72')
    axes[0].set(xticks=x,xticklabels=DATES,ylabel='% видимых точек крыши',ylim=(0,105),title='Контрольная крыша в LAS');axes[0].legend(fontsize=8)
    axes[1].bar(x,[float(rec(d,'source','roof','fusion')['built_fraction'])*100 for d in DATES],color='#2c8a72')
    axes[1].set(xticks=x,xticklabels=DATES,ylabel='% видимых точек крыши',ylim=(0,105),title='Та же крыша в 3DGS: пока 0% на всех датах')
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(OUT/'recognition.png',dpi=150);plt.close(fig)
    for d in DATES:
        fig,axes=plt.subplots(2,3,figsize=(14,9.5),constrained_layout=True)
        for row,side in enumerate(['source','target']):
            for col,(folder,suffix,title) in enumerate([(OUT,'rgb','Изображение'),(ROOT/'results/semantic_matching','semantic','Прежние классы'),(OUT,'semantic','Новые классы')]):
                image=cv2.cvtColor(cv2.imread(str(folder/d/f'{side}_{suffix}.png')),cv2.COLOR_BGR2RGB)
                axes[row,col].imshow(image);axes[row,col].set_title(('3DGS' if side=='source' else 'LAS')+' · '+title);axes[row,col].axis('off')
        fig.suptitle(d+': прогнозы на видимых точках; белое — отсутствие точки, серое — неопределённость')
        fig.legend(handles=[Patch(color=color/255,label=name) for color,name in zip(COLORS,['Неуверенно','Конструкции','Покрытие','Грунт','Растительность','Вода','Подвижное'])],loc='lower center',ncol=7,fontsize=9)
        fig.savefig(OUT/d/'semantics.png',dpi=120);plt.close(fig)
    counts='; '.join(f'{name}: {len(accepted[m])}/7' for m,name in zip(METHODS,NAMES))
    lines=['# Семантика: распознавание конструкций и отказ 0723','',counts+'.','',
        f"Принятых улучшений медианы более 1 см к текущему методу: **{summary['new_accepted_median_improvements_over_1cm']}**, ухудшений: **{summary['new_accepted_median_regressions_over_1cm']}**. Этот порог описывает размер различия, не статистическую значимость. Рабочий комбинированный метод остаётся основным.",'',
        '## Сравнение всех семи дат','',
        '| Дата | Текущий, медиана / P90, м | Прежняя семантика, допуск | Прежние метки + банк пар, допуск | Новая семантика, медиана / P90, м | Новый допуск |',
        '|---|---:|---|---|---:|---|']
    for d in DATES:
        ok=lambda m:'да' if d in accepted[m] else 'отказ'
        lines.append(f"| {d} | {value(d,'current_automatic')} / {value(d,'current_automatic','p90')} | {ok('previous_semantic')} | {ok('old_semantic_bank')} | {value(d,'semantic_fusion')} / {value(d,'semantic_fusion','p90')} | {ok('semantic_fusion')} |")
    lines+=['','Допуск — прежний внутренний фильтр масштаба, геометрии, RGB и минимум трёх разнесённых подтверждённых пар. Отклонённые матрицы, если получены, остаются в CSV с явным статусом. Числа описывают расстояние центров гауссианов до данного LAS на одинаковых пространственно отложенных точках с прежней фиксированной областью покрытия; это не независимая геодезическая точность.','',
        'Текущий метод здесь — автоматический выбор из `object_recovery/selection.json`; он отличается от исторических полных экспортов просмотрщика. Ни текущая окончательная матрица, ни RTK, ни ручной старт 0708 не используются для выбора новых стартов.','',
        '## Что изменено','',
        '1. Добавлен специалист по зданиям: [SegFormer-B2, обученный на LoveDA](https://huggingface.co/wu-pr-gw/segformer-b2-finetuned-with-LoveDA), версия `5c74556c08bebb5f45f50b6f78f61a62c5d220c7`. [LoveDA](https://github.com/Junjue-Wang/LoveDA) содержит отдельный класс зданий на видах сверху. Конфигурация автора задаёт восемь выходов, включая Ignore/Background; используется её нормализация. Код из репозитория модели не исполняется; PyTorch читает только веса через `weights_only=True`, SHA-256 весов проверен.','',
        '2. Виды строятся по прежним обучающим точкам. Для изображения пробелы заполняются ближайшим наблюдаемым цветом и медианным фильтром 3×3. В источнике радиус связан с размером отложенной плитки 0,4 исходной единицы; в LAS — 24 пикселя. Это приближение контекста, а не восстановленная поверхность: заполненные пиксели никогда не становятся 3D-точками. Возле границ видны искусственные продолжения текстуры.','',
        '3. LoveDA применяется как специалист по зданиям при вероятности ≥0,6. Остальные классы сохраняет прежняя drone-модель. Её уверенные метки растительности, воды и подвижных объектов защищены от замены зданиями. Прежние метки конструкций заменяются новым прогнозом либо неопределённостью. Полная замена всех классов на LoveDA была отброшена из-за отсутствия достаточных общих классов в части пар облаков, до итогового семидатного сравнения. Сырые прогнозы всех дат сохранены в `semantic_structures`; диагностический запуск регистрации 0331 также сохранён.','',
        '4. Для каждого выбранного старта заранее объединяются пары, предложенные исходными гипотезами в окрестности RMS <2 м по тем же обучающим якорям. Каждая пара хранит номера предложивших её гипотез. После подгонки допускаются только эти исходные идентичности с прежним взаимно однозначным геометрическим назначением, двусторонним перекрытием и пространственным разнесением. Новые пары из окончательной позы не добавляются.','',
        'Все три новых варианта имеют одинаковый пул и бюджет четырёх стартов, одинаковые этапы ICP и финальной подгонки. Контроль «прежние метки + банк пар» отделяет исправление 0723 от новой нейросети. Отложенные ошибки не задают параметры. Это следующий опыт разработки на уже известной площадке, не новый независимый тест.','',
        '## Почему отказывал 0723','',
        'У прежних геометрического и семантического результатов максимальная разница элементов матрицы была 8,53×10⁻¹⁴: положение практически одинаково. Однако старт `recovery_366` сохранял пять пар, а соседний `recovery_373` — шесть. Среди прежних подтверждённых пар две находятся близко и не дают отдельных пространственных опор.','',
        'Банк соседних стартов 558 и 564 добавляет одну заранее предложенную пару `p8_c16 → p8_c39`. В диагностике старой окончательной позы она даёт третью разнесённую опору. Её доли точек ближе 1 м: 71,2% вперёд и 55,4% назад, расстояние центров 4,895 м при лимите 5 м. Это ограниченный запас, а не подтверждение высокой надёжности. `0723_failure_analysis.json` сохраняет все детали.','',
        'Строгая проверка минимум трёх пар с прямым перекрытием ≥80%: '+', '.join(f'{name} {len(summary["strict_dates"][m])}/7' for m,name in zip(METHODS,NAMES))+'. Новый вариант всё ещё не проходит строгую проверку на 0708 и 0723. Увеличение числа пар при прежней матрице не означает улучшения точности.','',
        '## Проверка распознавания','',
        '| Дата | Крыша LAS, старый → новый прогноз | Крыша 3DGS, новый прогноз | Ложные конструкции в растительности LAS |',
        '|---|---:|---:|---:|']
    for d in DATES:
        pct=lambda side,region,method:f"{float(rec(d,side,region,method)['built_fraction'])*100:.1f}%"
        lines.append(f"| {d} | {pct('target','roof','previous')} → {pct('target','roof','fusion')} | {pct('source','roof','fusion')} | {pct('target','vegetation','fusion')} |")
    lines+=['','![Распознавание контрольной крыши](recognition.png)','',
        'Это доля видимых точек в фиксированных диагностических объёмах, не IoU и не глобальная точность классификации. Использованы ранее проверенная крыша и отдельно осмотренная по RGB полоса растительности без строений. Числа точек и старые/новые метки для обоих облаков находятся в `recognition.csv`. Предыдущая автоматическая матрица используется только для расположения диагностических объёмов в источнике; эти объёмы не читаются сегментацией или регистрацией.','',
        '**Улучшение распознавания частичное:** крыша LAS распознаётся на пяти датах, но крыша 3DGS остаётся пропущенной на всех семи. Уверенных меток зданий недостаточно для общего класса на датах '+', '.join(d for d in DATES if d not in summary['shared_built_class_dates'])+'. Шумные центры гауссианов и отсутствующие плитки сильно отличаются от снимков, на которых обучена модель. Нужна проверенная разметка и адаптация к этим проекциям либо более качественные виды 3DGS; увеличение числа оранжевых точек не доказывает точности распознавания.','',
        '[RGB-проверка отрицательной области](negative_region_review.png). На одной крыше и одной полосе растительности нельзя установить полноту и точность распознавания всей стройплощадки. Модели не имеют отдельного класса кранов. Локальная проверка неподвижных конструкций использует прежние маски исключения техники.','',
        '## Неподвижные конструкции и воспроизведение','',
        'В `static_structures.csv` сохранены крыша и прибрежная конструкция: отдельные неиспользованные записи и строгие пространственные выборки. Первая выборка имеет соседей в обучении, вторая на узких конструкциях часто мала. Пустые метрики не заменены нулями. BIM и рабочие полные PLY не перемещаются.','',
        '```sh','OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python scripts/prepare_semantic_structures.py',
        'OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python scripts/prepare_semantic_fusion.py',
        'OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python scripts/align_semantic_fusion.py',
        '.venv/bin/python scripts/check_semantic_recognition.py','.venv/bin/python scripts/report_semantic_fusion.py',
        "OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -m unittest discover -s scripts -p 'test_*.py' -v > results/semantic_fusion/tests.log 2>&1",
        'OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python scripts/verify_semantic_fusion.py','```','',
        'Требуются сохранённые геометрические пулы и прежние drone-метки. Новых пакетов не добавлено. Веса, параметры и хеши кода зафиксированы до итогового сравнения. `validation.json` содержит численную перепроверку, `http_checks.json` — проверку локальных страниц. [Предыдущий опыт](../semantic_initial/REPORT.md) сохранён.','',
        'В файлах `*_labels.npz` поля `labels`, `confidence` и `label_origin` описывают итоговое решение. `probabilities` сохраняет исходные вероятности drone-модели, а `building_expert_probabilities` — прогноз LoveDA; это два отдельных распределения, не единая откалиброванная вероятность нового метода. `label_origin`: 0 — прежний класс, 1 — специалист по зданиям, 2 — прежняя метка здания заменена неопределённостью.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    page=['<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Конструкции и исправление 0723</title>',
        '<style>body{font:16px/1.6 system-ui,sans-serif;color:#17202a;max-width:1300px;margin:24px auto;padding:0 20px}p{max-width:85ch}img{width:100%;height:auto}a{color:#175794}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:8px;text-align:left;border-bottom:1px solid #ccd2d8}.table{overflow:auto}summary{cursor:pointer;font-weight:650;padding:14px 0}details{border-bottom:1px solid #ccd2d8}a:focus-visible,summary:focus-visible{outline:2px solid #175794}h1{font-size:clamp(24px,3vw,36px)}</style>',
        '<a href="/viewer/">← 3D-просмотрщик</a><h1>Конструкции и исправление 0723</h1>',
        f'<p>Основной фильтр: текущий метод {len(accepted["current_automatic"])}/7; прежняя семантика {len(accepted["previous_semantic"])}/7; прежние метки с банком пар {len(accepted["old_semantic_bank"])}/7; новый вариант {len(accepted["semantic_fusion"])}/7.</p>',
        '<p>На 0723 терялась одна удалённая опора при выборе между соседними стартами. Теперь их исходные пары сохраняются до подгонки и проходят прежние проверки. Это исправление соответствий отдельно от распознавания зданий.</p>',
        '<p>Строгий фильтр нового варианта: 5/7. На 0708 и 0723 он пока не пройден; результат 7/7 относится к основному фильтру.</p>',
        '<p><a href="REPORT.md">Полный отчёт</a> · <a href="summary.csv">Все варианты CSV</a> · <a href="recognition.csv">Распознавание CSV</a> · <a href="static_structures.csv">Неподвижные конструкции</a> · <a href="../semantic_initial/inspection.html">Предыдущий опыт</a></p>',
        '<div class="table"><table><thead><tr><th>Дата</th><th>Текущий: медиана / P90, м</th><th>Новый: медиана / P90, м</th><th>Новый допуск</th><th>Крыша LAS: было → стало</th></tr></thead><tbody>']
    for d in DATES:page.append(f'<tr><td>{d}</td><td>{value(d,"current_automatic")} / {value(d,"current_automatic","p90")}</td><td>{value(d,"semantic_fusion")} / {value(d,"semantic_fusion","p90")}</td><td>{"да" if d in accepted["semantic_fusion"] else "отказ"}</td><td>0% → {roof[DATES.index(d)]*100:.1f}%</td></tr>')
    page+=['</tbody></table></div>',
        '<p>Расстояния рассчитаны на одинаковых отложенных точках. Текущий комбинированный метод остаётся основным; общего улучшения точности новая семантика пока не подтверждает.</p>',
        '<img src="recognition.png" alt="Крыша LAS распознаётся на пяти датах, крыша 3DGS остаётся пропущенной на семи">',
        '<p><strong>Распознавание улучшено частично.</strong> LoveDA выделяет здания в LAS, прежняя модель сохраняет другие классы. Контрольная крыша 3DGS пока не распознаётся. Доли на одной крыше не являются точностью всей модели.</p>']
    for d in DATES:page.append(f'<details><summary>{d}: изображения и классы до / после</summary><a href="{d}/semantics.png"><img loading="lazy" src="{d}/semantics.png" alt="{d}: изображения, прежние и новые семантические классы"></a></details>')
    (OUT/'inspection.html').write_text('\n'.join(page)+'</html>\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
