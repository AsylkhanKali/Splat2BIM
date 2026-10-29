"""Compare frozen semantic reranking with the current and matched geometric methods."""
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from PIL import Image

from align_dates import ROOT,DATES
from check_las_vertical import write_csv
from check_static_structures import volume_mask
from semantic_projection import GROUPS,COLORS,PARAMETERS
from prepare_semantic_clouds import OUT,MODEL

METHODS=['current_automatic','pooled_geometry','semantic_rerank']
NAMES=['Текущий автоматический','Та же группа кандидатов, геометрия','Та же группа кандидатов, семантика']
RU=['Неуверенно / прочее','Конструкции','Покрытие','Грунт','Растительность','Вода','Подвижные объекты']


def main():
    audits={d:json.loads((OUT/d/'audit.json').read_text()) for d in DATES}
    rows=[r for d in DATES for r in audits[d]['comparison']];write_csv(OUT/'comparison.csv',rows)
    statics=[]
    for d in DATES:
        with (OUT/d/'static_structures.csv').open() as f:statics.extend(csv.DictReader(f))
    write_csv(OUT/'static_structures.csv',statics)
    def metric(date,method):return next((r for r in rows if r['date']==date and r['method']==method and r['scope']=='fixed_baseline_coverage'),None)
    def run(date,method):return next((r for r in audits[date]['runs'] if r['method']==method),None)
    def number(row,key):return '—' if row is None or row.get(key) is None else f"{row[key]:.3f}"
    summary_rows=[];coverage=[]
    roof=next(r for r in json.loads((ROOT/'results/static_structures/regions.json').read_text())['regions'] if r['id']=='roof')
    for date in DATES:
        old,geo,sem=[metric(date,m) for m in METHODS];a=audits[date]
        summary_rows.append({'date':date,'current_median_m':old['median'],'current_p90_m':old['p90'],
            'geometry_median_m':geo['median'] if geo else None,'semantic_median_m':sem['median'] if sem else None,
            'semantic_p90_m':sem['p90'] if sem else None,
            'semantic_minus_current_m':sem['median']-old['median'] if sem else None,
            'semantic_minus_geometry_m':sem['median']-geo['median'] if sem and geo else None,
            'semantic_status':run(date,'semantic_rerank')['status'],
            'geometry_candidate':run(date,'pooled_geometry').get('selected_candidate_id'),
            'semantic_candidate':run(date,'semantic_rerank').get('selected_candidate_id')})
        fig,axes=plt.subplots(2,2,figsize=(12,10),constrained_layout=True)
        for row,side in enumerate(['source','target']):
            for col,kind in enumerate(['rgb','semantic']):
                axes[row,col].imshow(Image.open(OUT/date/f'{side}_{kind}.png'));axes[row,col].axis('off')
                axes[row,col].set_title(f"{'3DGS, только обучающие точки' if side=='source' else 'LAS'} · {'цвет' if kind=='rgb' else 'прогноз модели'}")
        fig.suptitle(f'{date}: независимые проекции до совмещения',fontsize=15)
        fig.legend(handles=[Patch(color=c/255,label=n) for c,n in zip(COLORS,RU)],loc='outside lower center',ncol=4,fontsize=9)
        fig.savefig(OUT/date/'semantics.png',dpi=125);plt.close(fig)
        for side in ['source','target']:
            data=np.load(OUT/date/f'{side}_labels.npz');labels=data['labels'];counts=np.bincount(labels,minlength=len(GROUPS))
            row={'date':date,'side':side,'visible_points':len(labels),'confident_static_fraction':float(np.isin(labels,PARAMETERS['static_groups']).mean()),
                **{name:int(counts[i]) for i,name in enumerate(GROUPS)},
                'reviewed_roof_visible_points':None,'reviewed_roof_predicted_built_fraction':None}
            if side=='target':
                mask=volume_mask(data['points'],roof);row['reviewed_roof_visible_points']=int(mask.sum())
                row['reviewed_roof_predicted_built_fraction']=float(np.mean(labels[mask]==1)) if mask.any() else None
            coverage.append(row)
    write_csv(OUT/'summary.csv',summary_rows);write_csv(OUT/'semantic_coverage.csv',coverage)
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),constrained_layout=True);x=np.arange(len(DATES));width=.25
    for i,(method,name,color) in enumerate(zip(METHODS,NAMES,['#5d79aa','#bd7538','#269273'])):
        for ax,key,title in zip(axes,['median','p90'],['Медиана расстояния до LAS','90-й процентиль расстояния до LAS']):
            values=[metric(d,method)[key] if metric(d,method) else np.nan for d in DATES]
            ax.bar(x+(i-1)*width,values,width,label=name,color=color)
            ax.set(xticks=x,xticklabels=DATES,title=title,ylabel='м');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(fontsize=8);fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
    accepted={m:sum((audits[d]['current_status'] if m==METHODS[0] else run(d,m)['status'])=='candidate_for_review' for d in DATES) for m in METHODS}
    strict={m:sum((audits[d]['current_strict_support'] if m==METHODS[0] else run(d,m).get('strict_forward_080_spatial_support_count',0))>=3 for d in DATES) for m in METHODS}
    comparable=[r for r in summary_rows if r['semantic_minus_current_m'] is not None]
    wins=sum(r['semantic_minus_current_m']<-.01 for r in comparable);losses=sum(r['semantic_minus_current_m']>.01 for r in comparable)
    changed=sum(r['geometry_candidate']!=r['semantic_candidate'] for r in summary_rows)
    summary={'accepted_dates':accepted,'strict_support_dates':strict,'semantic_improvements_over_1cm':wins,
        'semantic_regressions_over_1cm':losses,'within_1cm':len(comparable)-wins-losses,
        'changed_candidate_vs_matched_geometry':changed,
        'mean_date_medians_m':{m:float(np.mean([metric(d,m)['median'] for d in DATES if metric(d,m)])) for m in METHODS}}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Семантическое сопоставление: выбор после coarse-ICP','',
        'Это первый опыт. Также завершён [выбор стартов до ICP](../semantic_initial/REPORT.md) на полном исходном наборе гипотез; там сохранены и успешные, и отклонённые результаты.','',
        f"Использована предобученная нейросеть SegFormer-B0 для изображений с дрона. Семантика изменила выбор кандидата относительно одинакового геометрического контроля в {changed} из 7 дат. Относительно текущего автоматического метода улучшений медианы более 1 см: **{wins}**, ухудшений: **{losses}**, различий в пределах 1 см: **{len(comparable)-wins-losses}**. Это порог для описания величины различия, не тест статистической значимости.",'',
        f"Основной геометрический фильтр: текущий метод **{accepted[METHODS[0]]}/7**, общий набор кандидатов без семантики **{accepted[METHODS[1]]}/7**, с семантикой **{accepted[METHODS[2]]}/7**. Строгая проверка соответствий: {strict[METHODS[0]]}/7, {strict[METHODS[1]]}/7 и {strict[METHODS[2]]}/7 соответственно.",'',
        '## Что именно сравнивается','',
        '1. Текущий автоматический метод: сохранённое правило выбора из `object_recovery/selection.json`, без ручного старта и RTK. Это отдельный эксперимент; рабочие полные PLY просмотрщика используют ранее проверенный baseline.',
        '2. Геометрический контроль: общий набор из четырёх coarse-кандидатов прежнего консенсуса и четырёх восстановления масштаба. Выбор по геометрической медиане среди кандидатов с допустимым масштабом и ≥3 разнесёнными соответствиями.',
        '3. Семантический вариант: тот же набор и те же геометрические условия, но выбор по `геометрическая медиана + 0,5 × семантический балл`. Финальная геометрическая подгонка в вариантах 2 и 3 одинакова.','',
        'Семантический балл строится по ближайшим 3D-точкам **того же предсказанного класса**: конструкции, покрытие, грунт, растительность. В каждой общей группе нужно ≥128 видимых точек в обеих сценах; используется до 512 исходных точек на класс. Веса классов равны. Балл класса — 60-й процентиль расстояния с ограничением 20 м. Нужны минимум два общих класса. Вода, подвижные объекты, неопределённые метки не участвуют. Недостаток семантических данных даёт явный отказ, а не скрытую подстановку геометрического результата.','',
        'Это проверка семантического выбора из геометрически подготовленных гипотез. Сеть не создаёт новую позу с нуля, а набор ограничен восемью кандидатами, ранее отобранными геометрией. Прежние окончательные матрицы используются только в сравнении после выбора и подгонки новых результатов.','',
        '## Результаты на одинаковых отложенных точках','',
        '| Дата | Текущий: медиана / P90, м | Общие кандидаты, геометрия: медиана, м | Семантика: медиана / P90, м | Δ медианы к текущему, м | Решение семантики |',
        '|---|---:|---:|---:|---:|---|']
    for r in summary_rows:
        lines.append(f"| {r['date']} | {r['current_median_m']:.3f} / {r['current_p90_m']:.3f} | {number(r,'geometry_median_m')} | {number(r,'semantic_median_m')} / {number(r,'semantic_p90_m')} | {number(r,'semantic_minus_current_m')} | {'кандидат' if r['semantic_status']=='candidate_for_review' else 'отказ'} |")
    lines+=['','![Сравнение медианы и P90](comparison.png)','',
        'Все расстояния рассчитаны на прежних пространственных плитках, исключённых из подгонки и семантического рендеринга 3DGS. Таблица использует прежнюю фиксированную область покрытия LAS, одинаковую для всех методов. В `comparison.csv` также сохранены все отложенные точки без ограничения покрытия. Ни отложенные ошибки, ни текущая матрица не выбирают новый вариант. Различия отражают согласованность с LAS, а не независимую геодезическую точность.','',
        '## Качество семантического входа','',
        'Из каждого облака независимо строится вид сверху: вертикаль берётся из ранее оценённой обучающей плоскости, направление в плоскости — PCA. Общая система координат и готовая матрица для рендеринга не используются. Для 3DGS берутся только прежние обучающие записи. В каждом пикселе остаётся ближайшая сверху реальная точка; пустые пиксели не превращаются в 3D-соответствия. Мелкие пробелы до 4 пикселей заполняются цветом соседних обучающих точек только для изображения.','',
        'Сеть применяется к плиткам 512 × 512 с четырьмя поворотами на 90°. Усредняются вероятности; уверенность ниже 0,6 считается неопределённой. Предобработка точно взята из конфигурации модели: масштабирование RGB в [0,1], без ImageNet-нормализации. На этой площадке сеть не дообучалась.','',
        '| Дата | Уверенные статические метки 3DGS | Уверенные статические метки LAS | Точек 3DGS класса «конструкции» | Точек LAS класса «конструкции» |',
        '|---|---:|---:|---:|---:|']
    for d in DATES:
        s=next(r for r in coverage if r['date']==d and r['side']=='source');t=next(r for r in coverage if r['date']==d and r['side']=='target')
        lines.append(f"| {d} | {s['confident_static_fraction']:.1%} | {t['confident_static_fraction']:.1%} | {s['built']} | {t['built']} |")
    lines+=['','Знаменатель — видимые точки растровой проекции, не все гауссианы исходного файла. Эти доли не являются точностью классификации. На проекциях 3DGS очень мало уверенных меток конструкций; семантическое сопоставление в основном опирается на грунт и растительность. Фотографическая текстура LAS, шум центров гауссианов и пропуски обучающей выборки дают заметно разные изображения для одной сети. Размеченного независимого семантического теста здесь нет.','',
        'Неподвижные крыша и прибрежная конструкция проверены дополнительно на **тех же** сохранённых точках, что и текущий метод. Результаты всех трёх вариантов находятся в `static_structures.csv`. У выборки отдельных записей есть пространственные соседи в обучении; строгие пространственные выборки малочисленны. Пустая выборка отмечена пустыми метриками, а не нулём. Сеть не имеет класса «кран», поэтому её метки не заменяют ранее проверенные маски исключения техники.','']
    for d in DATES:lines+=[f'### {d}: прогнозы классов','',f'![Цвет и семантика {d}]({d}/semantics.png)','']
    lines+=['## Воспроизведение и происхождение модели','',
        '[Модель автора](https://huggingface.co/Thalirajesh/Aerial-Drone-Image-Segmentation) основана на SegFormer-B0 и содержит 24 исходных класса. Версия `fb5d31d4a149b819fade8d79288a9b48f7fa804f`; веса safetensors проверены по опубликованному SHA-256. Модель, её карточка, конфигурация и контрольные суммы сохранены в `models/drone_segformer_b0`. В метаданных автора лицензия обозначена как `other`; публичная передача весов и коммерческая пригодность этим опытом не устанавливаются. [Исходная архитектура](https://github.com/NVlabs/SegFormer).','',
        '```sh','OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 .venv/bin/python scripts/prepare_semantic_clouds.py',
        'OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python scripts/align_semantic.py',
        '.venv/bin/python scripts/report_semantic.py',
        "OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -m unittest discover -s scripts -p 'test_*.py' -v > results/semantic_matching/tests.log 2>&1",
        'OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python scripts/verify_semantic.py','```','',
        'Параметры и хеши кода зафиксированы в `segmentation_protocol.json` и `alignment_protocol.json` до сравнения. Веса и обработка локальны; проектные изображения никуда не загружаются. Численные проверки сохранены в `validation.json`. Эксперимент разработан на уже известной площадке и не заменяет проверку переноса на новую площадку.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    head='<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Семантическое сопоставление: семь дат</title>'
    style='<style>body{font:16px/1.6 system-ui,sans-serif;color:#17202a;background:white;max-width:1300px;margin:24px auto;padding:0 20px}p{max-width:80ch}img{width:100%;height:auto}a{color:#175794}a:focus-visible,summary:focus-visible{outline:2px solid #175794}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:8px;text-align:left;border-bottom:1px solid #ccd2d8}summary{cursor:pointer;font-weight:650;padding:14px 0}details{border-bottom:1px solid #ccd2d8}.table{overflow:auto}h1{font-size:clamp(24px,3vw,36px)}</style>'
    page=[head,style,'<a href="/viewer/">← 3D-просмотрщик</a><h1>Семантическое сопоставление: семь дат</h1>',
        '<p>Первый опыт: выбор после coarse-ICP. <a href="../semantic_initial/inspection.html">Перейти к сравнению выбора стартов до ICP</a>.</p>',
        f'<p>Изменён выбор кандидата: {changed}/7. Разница медианы к текущему методу более 1 см: улучшений {wins}, ухудшений {losses}. Остальные {len(comparable)-wins-losses} дат отличаются в пределах 1 см.</p>',
        f'<p>Геометрический фильтр прошли: текущий метод {accepted[METHODS[0]]}/7; общий набор кандидатов без семантики {accepted[METHODS[1]]}/7; с семантикой {accepted[METHODS[2]]}/7.</p>',
        '<p>Сеть распознаёт классы на независимых проекциях облаков. На 3DGS уверенных меток зданий мало; основную опору дают грунт и растительность. Числа ниже показывают согласованность с LAS на отложенных точках.</p>',
        '<p><a href="REPORT.md">Методика и ограничения</a> · <a href="summary.csv">Сводка CSV</a> · <a href="static_structures.csv">Неподвижные конструкции</a></p>',
        '<div class="table"><table><thead><tr><th>Дата</th><th>Текущий, м</th><th>Геометрия, м</th><th>Семантика, м</th><th>Δ, м</th></tr></thead><tbody>']
    for r in summary_rows:page.append(f"<tr><td>{r['date']}</td><td>{r['current_median_m']:.3f}</td><td>{number(r,'geometry_median_m')}</td><td>{number(r,'semantic_median_m')}</td><td>{number(r,'semantic_minus_current_m')}</td></tr>")
    page+=['</tbody></table></div><p>В таблице указаны медианы расстояния до LAS.</p><img src="comparison.png" alt="Сравнение медианы и P90 для трёх методов на семи датах">']
    for d in DATES:page.append(f'<details><summary>{d}: посмотреть цвет и семантические классы</summary><a href="{d}/semantics.png"><img loading="lazy" src="{d}/semantics.png" alt="{d}: независимые проекции 3DGS и LAS и прогнозы классов"></a></details>')
    (OUT/'inspection.html').write_text('\n'.join(page)+'</html>\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
