"""Compare seven Gaussian recognition runs and explain unchanged strict failures."""
import json

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from align_dates import ROOT,DATES
from align_0929 import apply
from check_static_structures import volume_mask
from check_las_vertical import write_csv,file_hash
from prepare_semantic_3dgs import OUT,OLD,PREVIOUS


def main():
    region_path=PREVIOUS/'recognition_regions.json';regions=json.loads(region_path.read_text())['regions']
    rows=[];summary=[];negative_fig,negative_axes=plt.subplots(1,7,figsize=(16,4),constrained_layout=True)
    for date,negative_ax in zip(DATES,negative_axes):
        folder=OUT/date;new=dict(np.load(folder/'source_labels.npz'));old=dict(np.load(PREVIOUS/date/'source_labels.npz'))
        matrix_path=ROOT/f'results/object_recovery/{date}/selection.json';matrix=np.array(json.loads(matrix_path.read_text())['matrix_local'])
        world=apply(new['points'],matrix);masks={r['id']:volume_mask(world,r) for r in regions}
        for method,data in [('previous_fusion',old),('gaussian_render',new)]:
            for region,mask in {**masks,'all_visible':np.ones(len(world),bool)}.items():
                rows.append({'date':date,'method':method,'region':region,'count':int(mask.sum()),
                    'built_points':int(np.sum(data['labels'][mask]==1)),
                    'built_fraction':float(np.mean(data['labels'][mask]==1)) if mask.any() else None})
        previous_audit=json.loads((PREVIOUS/date/'audit.json').read_text());run=next(r for r in previous_audit['runs'] if r['method']=='semantic_fusion')
        diagnostic_path=ROOT/f'results/strict_diagnostics/{date}.json';diagnostic=json.loads(diagnostic_path.read_text())
        value=lambda method,region:next(r['built_fraction'] for r in rows if r['date']==date and r['method']==method and r['region']==region)
        summary.append({'date':date,'roof_count':int(masks['roof'].sum()),'previous_roof_fraction':value('previous_fusion','roof'),
            'new_roof_fraction':value('gaussian_render','roof'),'vegetation_count':int(masks['vegetation'].sum()),
            'previous_vegetation_fraction':value('previous_fusion','vegetation'),'new_vegetation_fraction':value('gaussian_render','vegetation'),
            'old_built_count':int(np.sum(old['labels']==1)),'new_built_count':int(np.sum(new['labels']==1)),
            'strict_overlap_pairs':diagnostic['strict_pair_count'],'admissible_triangles':diagnostic['admissible_bank_triangles'],
            'previous_primary_status':run['status'],'previous_strict_pass':run['strict_forward_080_spatial_support_count']>=3,
            'evaluation_matrix_sha256':file_hash(matrix_path),'strict_diagnostic_sha256':file_hash(diagnostic_path)})
        old_rgb=cv2.cvtColor(cv2.imread(str(PREVIOUS/date/'source_rgb.png')),cv2.COLOR_BGR2RGB)
        new_rgb=cv2.cvtColor(cv2.imread(str(folder/'source_rgb.png')),cv2.COLOR_BGR2RGB)
        overlay=cv2.cvtColor(cv2.imread(str(folder/'source_built.png')),cv2.COLOR_BGR2RGB)
        uv=new['render_uv'];roof=uv[masks['roof']];lo=np.maximum(roof.min(0)-40,0);hi=np.minimum(roof.max(0)+41,2048)
        fig,axes=plt.subplots(2,3,figsize=(14,9),constrained_layout=True)
        for col,(image,title) in enumerate([(old_rgb,'Прежние центры и заполнение пробелов'),(new_rgb,'Гауссианы: форма, прозрачность, SH'),(overlay,'Новые метки конструкций: оранжевый')]):
            axes[0,col].imshow(image);axes[0,col].set_title(title,fontsize=10);axes[0,col].axis('off')
            factor=image.shape[0]/2048;a=np.floor(lo*factor).astype(int);b=np.ceil(hi*factor).astype(int)
            axes[1,col].imshow(image[a[1]:b[1],a[0]:b[0]]);axes[1,col].axis('off')
        fig.suptitle(f"{date} · контрольная крыша: {value('previous_fusion','roof')*100:.1f}% → {value('gaussian_render','roof')*100:.1f}%\nНижний ряд: одинаковая область, выделенная только для проверки")
        fig.savefig(folder/'comparison.png',dpi=130);plt.close(fig)
        neg=uv[masks['vegetation']]
        if len(neg):
            lo=np.maximum(neg.min(0)-15,0);hi=np.minimum(neg.max(0)+16,2048)
            negative_ax.imshow(new_rgb[lo[1]:hi[1],lo[0]:hi[0]])
        negative_ax.set_title(date);negative_ax.axis('off')
    negative_fig.suptitle('3DGS RGB: фиксированная отрицательная область растительности, без меток модели')
    negative_fig.savefig(OUT/'negative_region_review.png',dpi=150);plt.close(negative_fig)
    write_csv(OUT/'recognition.csv',rows);write_csv(OUT/'summary.csv',summary)
    result={'registration_rerun':False,'recognition_regions_sha256':file_hash(region_path),'dates':summary,
        'roof_detected_dates':[r['date'] for r in summary if r['new_roof_fraction']>0],
        'unchanged_strict_pass_dates':[r['date'] for r in summary if r['previous_strict_pass']],
        'max_vegetation_false_built_fraction':max(r['new_vegetation_fraction'] for r in summary if r['new_vegetation_fraction'] is not None)}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True);x=np.arange(7)
    for ax,region,title in zip(axes,['roof','vegetation'],['Контрольная крыша 3DGS','Ложные конструкции: растительность 3DGS']):
        for shift,method,label,color in [(-.18,'previous_fusion','Прежняя проекция','#78879c'),(.18,'gaussian_render','Проекция гауссианов','#21876f')]:
            values=[next(r['built_fraction'] for r in rows if r['date']==d and r['method']==method and r['region']==region) for d in DATES]
            ax.bar(x+shift,np.array(values)*100,.36,label=label,color=color)
        ax.set(xticks=x,xticklabels=DATES,ylim=(0,105),ylabel='% видимых точек',title=title);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(OUT/'recognition.png',dpi=150);plt.close(fig)
    table=['| Дата | Точек крыши | Крыша: было → стало | Ложные здания в растительности | Пар с достаточным перекрытием | Строгая проверка прежней позы |','|---|---:|---:|---:|---:|---|']
    for r in summary:table.append(f"| {r['date']} | {r['roof_count']} | {r['previous_roof_fraction']*100:.1f}% → {r['new_roof_fraction']*100:.1f}% | {r['new_vegetation_fraction']*100:.1f}% | {r['strict_overlap_pairs']} | {'пройдена' if r['previous_strict_pass'] else 'не пройдена'} |")
    failure_0708='0708: три пары имеют нужное перекрытие, но их центры почти лежат на одной линии. Вторая сингулярная величина центрированных координат источника равна 1,962274 м при минимуме 2 м; для LAS — 4,353667 м. Минимальное расстояние между опорами около 59 м, поэтому проблема именно в поперечном разнесении. Отступ до порога — 3,77 см по этой метрике, не ошибка положения объекта. Порог не снижен.'
    failure_0723='0723: только две пары удовлетворяют прямому перекрытию ≥80% и обратному ≥50% в пределах 1 м. Третья разнесённая пара p8_c16 → p8_c39 даёт 71,15% / 55,38%: 37 из 52 исходных точек ближе 1 м к назначенному фрагменту, нужно минимум 42. Это нехватка пяти подтверждённых точек, а не предложение добавить искусственные точки.'
    note='В этом опыте улучшено распознавание конструкций в 3DGS. Строгие статусы относятся к прежнему семантическому совмещению: его преобразования и геометрические опоры сохранены. Улучшение распознавания само по себе не доказывает улучшения совмещения.'
    lines=['# 3DGS: распознавание конструкций и строгие отказы','',note,'',
        f"Контрольная крыша получила метки конструкций на {len(result['roof_detected_dates'])}/7 датах вместо 0/7. Это доля выбранных видимых точек в одном объёме, не IoU и не точность всей модели.",'',*table,'','![Сравнение](recognition.png)','',
        '## Что изменено','',
        'Прежние разреженные центры с заполненными цветом пробелами заменены ортографическим рендером 3D-гауссианов: проецируются ковариации из масштаба и кватерниона, используется прозрачность с композитингом от дальних к ближним, цвет вычисляется со сферическими гармониками до степени 3. Разрешение увеличено с 1024 до 2048, границы изображения сохранены. Это ортографический вид, а не точное воспроизведение камеры исходного фото.','',
        'Модель LoveDA B2, нормализация, плитки 512 пикселей с четырьмя поворотами, порог зданий 0,6 и защита уверенных drone-классов растительности, воды и подвижных объектов сохранены. Новые классы присвоены ровно прежним видимым 3D-точкам; заполненные пиксели не создают геометрию. LAS-метки берутся из прежнего опыта.','',
        'Для изображения используются только чётные исходные записи PLY, внутри прежних обучающих границ и вне пространственно отложенных плиток 0,4 исходной единицы. Прозрачные гауссианы с alpha >1/255 сохранены для контекста. Их отпечатки могут пересекать границы исключённых плиток, но центры контрольных гауссианов не читаются рендером. Ни окончательная матрица, ни BIM, ни диагностические объёмы не задают проекцию или прогноз.','',
        '**Граница проверки:** разрешение выбрано после исследовательских проб на крыше 0723 и 0929; это этап разработки на известной площадке. Одинаковая конфигурация затем применена ко всем семи датам. Результаты на крыше не являются независимой тестовой оценкой. Изменились одновременно способ изображения и разрешение; их отдельный вклад здесь не измерен. Предыдущая автоматическая матрица используется только для нахождения диагностических объёмов.','',
        '[RGB отрицательной области на семи датах](negative_region_review.png). Полоса растительности проверяет один тип ложных срабатываний. Она не оценивает краны, технику, грунт или всю площадку; отдельного класса кранов эта модель не имеет.','',
        '## Почему строгая проверка не пройдена','',failure_0708,'',failure_0723,'',
        'На обеих датах перебраны все тройки подходящих пар: допустимой тройки нет. Дополнительное сопоставление всех фрагментов в окончательной позе также не нашло её; оно выполнено только для диагностики и не использовано для принятия результата.','',
        'Все точки третьей пары 0723 находятся в пределах 1 м от полного LAS, но не все близки к назначенному фрагменту. Это указывает на различия границ/идентичности фрагментов: близость к любому соседу в LAS не подтверждает правильное соответствие объекта. На 0708 у пары p9_c36 → p6_c2 обратное перекрытие лишь 44,52%; 87,2% обратных промахов находятся в отложенных плитках источника. Исключать эти промахи из проверки ради допуска нельзя.','',
        'Следующий обоснованный шаг для строгого допуска: на 0708 подтвердить устойчивую опору вне почти прямой линии, на 0723 выделить полные соответствующие части одного объекта. Нужны проверяемые идентичности конструкций, заданные до подгонки; подмена пар после результата или ослабление порогов не применены.','',
        '![0708](../strict_diagnostics/0708_diagnostic.png)','![0723](../strict_diagnostics/0723_diagnostic.png)','',
        '## Файлы и воспроизведение','',
        '`summary.csv` — сравнение семи дат; `recognition.csv` — количества в контрольных областях; `<дата>/source_labels.npz` — новые метки исходных 3D-точек и отдельные вероятности LoveDA; `render_context.npz` — исходные индексы контекста; `segmentation_protocol.json` — параметры и хеши; `../strict_diagnostics/` — пары, тройки и пороги. Поле probabilities остаётся прежним распределением drone-модели, building_expert_probabilities хранит новый прогноз LoveDA, labels/confidence/label_origin — итог слияния.','',
        '```sh','.venv/bin/python scripts/prepare_semantic_3dgs.py','.venv/bin/python scripts/diagnose_strict_support.py','.venv/bin/python scripts/report_semantic_3dgs.py',
        ".venv/bin/python -m unittest discover -s scripts -p 'test_*.py' -v > results/semantic_3dgs/tests.log 2>&1",
        '.venv/bin/python scripts/verify_semantic_3dgs.py','```','',
        'Используются существующие пакеты и веса; нужен clang++ для локального компилятора рендера (на этой машине уже установлен). Предыдущие результаты и исходные assets сохранены. Детали выполненных проверок — в validation.json. [Прежнее сравнение регистрации](../semantic_fusion/REPORT.md).','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    page=['<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>3DGS: конструкции и строгие проверки</title>',
        '<style>body{font:16px/1.6 system-ui,sans-serif;color:#17202a;max-width:1300px;margin:24px auto;padding:0 20px}p{max-width:90ch}img{width:100%;height:auto}a{color:#175794}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:9px;text-align:left;border-bottom:1px solid #ccd2d8}.table{overflow:auto}summary{cursor:pointer;font-weight:650;padding:14px 0}details{border-bottom:1px solid #ccd2d8}a:focus-visible,summary:focus-visible{outline:2px solid #175794}h1{font-size:clamp(24px,3vw,36px)}</style>',
        '<a href="/viewer/">← 3D-просмотрщик</a><h1>3DGS: конструкции и строгие проверки</h1>',
        f'<p>Контрольная крыша распознаётся на <strong>{len(result["roof_detected_dates"])}/7 датах</strong> вместо 0/7. Новая проекция учитывает форму и прозрачность гауссианов. Модель и порог уверенности сохранены.</p>',
        '<p>'+note+'</p>',
        '<p><a href="REPORT.md">Методика и ограничения</a> · <a href="summary.csv">Сравнение CSV</a> · <a href="recognition.csv">Количество точек</a> · <a href="../semantic_fusion/inspection.html">Прежнее сравнение регистрации</a></p>',
        '<div class="table"><table><thead><tr><th>Дата</th><th>Точек крыши</th><th>Распознано: было → стало</th><th>Ложные здания в растительности</th><th>Строгая проверка прежней позы</th></tr></thead><tbody>']
    for r in summary:page.append(f'<tr><td>{r["date"]}</td><td>{r["roof_count"]}</td><td>{r["previous_roof_fraction"]*100:.1f}% → {r["new_roof_fraction"]*100:.1f}%</td><td>{r["new_vegetation_fraction"]*100:.1f}%</td><td>{"Пройдена" if r["previous_strict_pass"] else "Не пройдена"}</td></tr>')
    page+=['</tbody></table></div><p>Доли относятся к одной крыше и одной полосе растительности. Настройки разработаны на этой площадке; это не независимая оценка всей сцены.</p>',
        '<img src="recognition.png" alt="Распознавание контрольной крыши и ложные конструкции в растительности на семи датах">',
        '<h2>Причины строгих отказов</h2><p>'+failure_0708+'</p><p>'+failure_0723+'</p>',
        '<p>Проверены все тройки: подходящей нет. Пороги не ослаблены. Основной допуск прежнего метода остаётся 7/7, строгий — 5/7.</p>']
    for d in ['0708','0723']:page.append(f'<details><summary>{d}: опоры, перекрытие и полный LAS</summary><img loading="lazy" src="../strict_diagnostics/{d}_diagnostic.png" alt="{d}: диагностика строгого отказа"></details>')
    page.append('<h2>Изображения всех семи дат</h2><p>Сверху — вся сцена; снизу — контрольная крыша. Оранжевые точки — итоговые метки конструкций.</p>')
    for d in DATES:page.append(f'<details><summary>{d}: прежняя проекция → гауссианы → распознанные конструкции</summary><a href="{d}/comparison.png"><img loading="lazy" src="{d}/comparison.png" alt="{d}: сравнение проекций и меток конструкций"></a></details>')
    page.append('<details><summary>Проверка отрицательной области растительности</summary><img loading="lazy" src="negative_region_review.png" alt="Виды фиксированной полосы растительности в 3DGS на семи датах"></details>')
    (OUT/'inspection.html').write_text('\n'.join(page)+'</html>\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
