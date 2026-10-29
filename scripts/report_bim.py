"""Summarize the unadjusted BIM check and populate viewer inspection points."""
from pathlib import Path
import csv
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/0929_bim'


def main():
    versions={v:json.loads((OUT/f'{v}_elements.json').read_text()) for v in ('v1','v2','v3')}
    manifest=json.loads((OUT/'manifest.json').read_text())
    elements=versions['v3']['elements']
    groups={name:[e for e in elements if name.lower() in e['name'].lower()] for name in ('Deck','Canopy','Pier','Pile Cap','Kiosk')}
    groups['Underground piles']=[e for e in elements if e['type']=='IfcPile']
    summary={}
    for name,items in groups.items():
        errors=np.array([e['median_surface_to_las_m'] for e in items])
        summary[name]={'elements':len(items),'median_of_element_medians_m':float(np.median(errors)),
                       'p90_of_element_medians_m':float(np.quantile(errors,.9))}
    changes={}
    current={e['guid']:e for e in elements}
    for v in ('v1','v2'):
        previous={e['guid']:e for e in versions[v]['elements']}
        common=current.keys()&previous.keys()
        changes[v]={'common':len(common),'same_geometry':sum(current[k]['geometry_sha256']==previous[k]['geometry_sha256'] for k in common),
                    'removed':len(previous.keys()-current.keys()),
                    'added':[current[k] for k in sorted(current.keys()-previous.keys())]}
    choices=[('9525','Настил у FP4/FP5'),('49665','Навес, северный участок'),('11021','Опора FP1'),
             ('10912','Ростверк — проверить видимость'),('5828','Киоск, западный участок')]
    by_id={e['id']:e for e in elements}
    manifest['landmarks']=[{'id':key,'label':label,'center':by_id[key]['center'],
         'bounds_min':by_id[key]['bounds_min'],'bounds_max':by_id[key]['bounds_max'],
         'note':f"BIM v3 #{key}; медиана до LAS {by_id[key]['median_surface_to_las_m']:.2f} м (500 проб поверхности)"}
         for key,label in choices]
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    result={'coordinate_adjustment':'Only subtract [825800,842100,0]; no rotation, scale, ICP or fitted translation applied to BIM.',
            'revision_for_inspection':'v3; authoritative revision not confirmed',
            'group_statistics':summary,'revision_comparison':changes,
            'landmarks':[by_id[key] for key,label in choices],
            'limitations':['Distances are 500 area-weighted random BIM surface samples to nearest LAS point per element, not corresponding survey control.',
                           'Entire surfaces include undersides, hidden faces and underground geometry; this is not a construction-completion score.',
                           'Element medians are aggregated with equal element weight, not by area.',
                           'Source-space sample seeds are IFC entity IDs; identical geometry may have slightly different sampled distances between revisions.']}
    (OUT/'bim_check.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    with (OUT/'v3_element_distances.csv').open('w',newline='') as stream:
        fields=['id','guid','type','name','median_surface_to_las_m','fraction_within_1m']
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(elements)
    lines=['# Проверка 0929 ↔ BIM','',
           '3DGS и все три версии BIM помещены в одну локальную систему метров. К BIM применён только тот же сдвиг начала координат `[825800, 842100, 0]`; дополнительной подгонки BIM не было. Общая привязка выглядит согласованной, но это предварительная проверка, а не геодезическая приёмка.','',
           '## Версии и полнота','',
           '| Версия | Преобразовано элементов | Не представлено сеткой |','|---|---:|---:|']
    for v,info in versions.items():lines.append(f"| {v} | {info['converted']} | {len(info['unconverted'])} |")
    lines+=['','Во всех трёх файлах исключён только `IFC Project Base Point` — аннотация, а не строительная поверхность.',
            'В v3 сохранена геометрия всех 5579 элементов v2 и добавлены 7 элементов: четыре киоска и три элемента без имени. Поэтому v3 используется для просмотра как наиболее полная из этих двух версий; её официальный статус нужно подтвердить.',
            'В v1 и v3 совпадают 5479 элементов с теми же GUID и геометрией; 200 элементов v1 отсутствуют в v3, а 107 добавлены. Версии не следует смешивать в одну модель.','',
            '## Расстояния BIM → LAS','',
            'На каждом элементе взято 500 проб с вероятностью, пропорциональной площади треугольников. В таблице ниже — медиана индивидуальных медиан, а не глобальная точность регистрации.','',
            '| Группа | Элементов | Медиана медиан, м | 90-й процентиль медиан, м |','|---|---:|---:|---:|']
    for name,row in summary.items():lines.append(f"| {name} | {row['elements']} | {row['median_of_element_medians_m']:.3f} | {row['p90_of_element_medians_m']:.3f} |")
    lines+=['','## Участки для визуального осмотра','',
            '| Элемент v3 | Медиана расстояния до LAS, м | Доля проб ближе 1 м |','|---|---:|---:|']
    for key,label in choices:
        e=by_id[key];lines.append(f"| {label} · #{key} | {e['median_surface_to_las_m']:.3f} | {e['fraction_within_1m']:.1%} |")
    lines+=['','Эти участки выбраны для осмотра, а не как независимая контрольная выборка. Полные поэлементные результаты сохранены в `v3_element_distances.csv`.',
            'Ростверки и сваи включают скрытые и подземные поверхности. Их большие расстояния до воздушной съёмки нельзя автоматически интерпретировать как смещение модели или отсутствие конструкции. Настилы также содержат нижние и боковые поверхности, которые видны не со всех ракурсов.','',
            '## Как открыть','',
            'В уже запущенном просмотрщике: http://127.0.0.1:8765/viewer/',
            'Переключатели показывают 3DGS, LAS, версии BIM; список участков приближает выбранные конструкции и отмечает их бирюзовой рамкой. Каркас намеренно включает скрытые части: это средство наложения, а не физически точная проверка видимости сквозь полупрозрачные гауссианы. Для чистого вида съёмки отключите слой BIM.','',
            'Для повторного запуска из корня проекта:','',
            '```sh','npm install --prefix viewer --ignore-scripts --no-audit --no-fund',
            '.venv/bin/python scripts/prepare_bim.py','.venv/bin/python scripts/report_bim.py',
            '.venv/bin/python scripts/serve_viewer.py','```','',
            'Просмотрщик использует локально установленные Spark 2.4.0 и Three.js 0.180.0. Сервер слушает только 127.0.0.1 и отдаёт только `viewer/` и `results/`; исходные данные не публикуются.']
    (OUT/'BIM_CHECK.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
