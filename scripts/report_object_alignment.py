"""Compare registration quality separately from correctness of proposed object pairs."""
import json

import laspy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from align_dates import ROOT,ORIGIN,DATES
from align_0929 import apply
from align_objects import OUT
from check_las_vertical import write_csv


def correspondence(folder,audit):
    run=next(r for r in audit['runs'] if r['method']=='objects');seed=run['selected_seed']
    source=np.load(folder/'source_candidate_points.npz')[seed['source_id']]
    target=np.load(folder/'target_candidate_points.npz')[seed['target_id']]
    moved=apply(source,np.array(run['matrix_local']));distance=cKDTree(target).query(moved)[0]
    result={'folder':folder.name,'date':audit['date'],'source_patch_count':len(source),'target_patch_count':len(target),
        'forward_median_m':float(np.median(distance)),'forward_p90_m':float(np.quantile(distance,.9)),
        'forward_fraction_within_1m':float(np.mean(distance<1.)),
        'centroid_separation_m':float(np.linalg.norm(moved.mean(0)-target.mean(0))),
        'registration_training_status':run['status']}
    result['pair_consistency_gate_passed']=result['forward_fraction_within_1m']>=.8
    result['review_status']='rejected_training_alignment' if run['status']=='rejected' else (
        'candidate_for_review' if result['pair_consistency_gate_passed'] else 'object_pair_not_confirmed')
    # Include both patches in the view even when the proposed association is wrong.
    las=laspy.read(ROOT/f'assets/{audit["date"]}/{audit["date"]}.las');p=np.c_[las.x,las.y,las.z]-ORIGIN
    color=np.c_[las.red,las.green,las.blue].astype(float);color/=65535 if color.max()>255 else 255
    all_points=np.r_[moved,target];lo=all_points[:,:2].min(0)-8;hi=all_points[:,:2].max(0)+8
    keep=np.all((p[:,:2]>=lo)&(p[:,:2]<=hi),axis=1);p=p[keep];color=color[keep];order=np.argsort(p[:,2])
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for ax,points,title in [(axes[0],moved,'Кандидат из 3DGS после совмещения'),(axes[1],target,'Выбранный объект LAS')]:
        ax.scatter(p[order,0],p[order,1],c=color[order],s=2)
        shown=points[::max(1,len(points)//500)]
        ax.scatter(shown[:,0],shown[:,1],s=12,facecolors='none',edgecolors='#de2458',linewidths=.8)
        ax.set(aspect='equal',xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),title=title,xlabel='X, м',ylabel='Y, м')
    fig.suptitle(f"{audit['date']} · медиана расстояния между выбранными фрагментами: {result['forward_median_m']:.2f} м")
    fig.savefig(folder/'correspondence_check.png',dpi=140);plt.close(fig)
    (folder/'correspondence_check.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    folders=[OUT/d for d in ['0929','0708']]+[OUT/(d+'_reference_roof') for d in DATES]
    records={};checks=[]
    for folder in folders:
        audit=json.loads((folder/'audit.json').read_text());records[folder.name]=audit
        if 'runs' in audit:checks.append(correspondence(folder,audit))
    write_csv(OUT/'correspondence_checks.csv',checks)
    all_rows=[]
    def metric(audit,method,scope='fixed_baseline_coverage'):
        return next(r for r in audit['comparison'] if r['method']==method and r['scope']==scope)
    for folder,audit in records.items():
        for row in audit.get('comparison',[]):all_rows.append({'folder':folder,**row})
    write_csv(OUT/'comparison.csv',all_rows)
    lines=['# Объектная начальная привязка 3DGS: первый эксперимент','',
        '27 октября 2026. Реализован поиск кандидатов на приподнятые светлые прямоугольные поверхности и получение начальной матрицы по парам таких кандидатов. После этого применяется геометрическое уточнение. Это прототип по правилам цвета и формы, **не обученная семантическая сегментация**.','',
        'Два режима: (а) искать кандидатов отдельно в 3DGS и LAS; (б) использовать одну ранее размеченную крышу в LAS, а кандидатов в 3DGS искать автоматически. Второй режим требует опорной области и не называется полностью автоматическим распознаванием сцены.','',
        '**Результат сохранённых прогонов:** в режиме с опорной крышей обе проверки проходит только 0929, дата разработки. На 0331, 0504 и 0609 нет совместимых гипотез; на 0723 совмещение отклонено; на 0708 и 0831 сцена совмещена, но выбранная объектная пара не подтверждена. На шести остальных датах успешный перенос всего метода пока не показан.','',
        'В новых преобразованиях не используются GPS, камеры COLMAP, прежние матрицы привязки или ручная начальная матрица 0708. Прежняя матрица читается после фиксации новых результатов — для сравнения и определения общей контрольной области. Рабочие экспортированные облака и BIM не заменены.','',
        '## Почему нужны две проверки','',
        'Проверяем отдельно: получилось ли совмещение всей сцены и действительно ли предложенная пара относится к одному объекту. ICP может исправить плохую объектную гипотезу и всё равно дать малую итоговую ошибку. Такой результат полезен как начальная гипотеза, но не доказывает распознавание объекта.','',
        '## Режим с опорной крышей: все семь дат','',
        'В каждой строке одинаковые отложенные записи исходного PLY. Основная метрика — расстояние центра гауссиана до ближайшей точки LAS, не независимая геодезическая точность. Показана фиксированная область покрытия прежней привязки; результаты по всем отложенным записям, включая неперекрывающиеся области, сохранены отдельно.','',
        '| Дата | Контрольных записей | PCA + ICP: медиана, м | Опорная крыша + ICP: медиана / P90, м | Рабочая привязка: медиана, м | Проверка выбранной пары |',
        '|---|---:|---:|---:|---:|---|']
    for date in DATES:
        name=date+'_reference_roof';a=records[name]
        if 'runs' not in a:
            lines.append(f'| {date} | — | — | — | — | Нет совместимых объектных гипотез |');continue
        p=metric(a,'pca');o=metric(a,'objects');b=metric(a,'existing_baseline');c=next(r for r in checks if r['folder']==name)
        status='геометрически согласована' if c['pair_consistency_gate_passed'] else '**не подтверждена**'
        if c['registration_training_status']=='rejected':status+='; совмещение отклонено'
        lines.append(f"| {date} | {o['count']:,} | {p['median']:.3f} | {o['median']:.3f} / {o['p90']:.3f} | {b['median']:.3f} | {status} |")
    lines+=['','У 0929 прежний результат использует RTK/COLMAP, у 0708 — документированную приблизительную ручную инициализацию. Это сильные рабочие ориентиры, но не те же входные условия, что у нового эксперимента. Они не выдаются за ещё один равнозначный вариант PCA.','',
        '## Полностью автоматический поиск кандидатов','',
        '| Дата | Медиана всей сцены, м | Медиана расстояния между выбранными фрагментами, м | Вывод о паре |',
        '|---|---:|---:|---|']
    for date in ['0929','0708']:
        a=records[date];o=metric(a,'objects');c=next(r for r in checks if r['folder']==date)
        lines.append(f"| {date} | {o['median']:.3f} | {c['forward_median_m']:.2f} | {'Согласована по геометрическому фильтру' if c['pair_consistency_gate_passed'] else 'Не подтверждена'} |")
    lines+=['','Это ограничивает вывод: удачная конечная матрица ещё не означает, что алгоритм правильно сопоставил крыши. В частности, автоматический вариант 0708 даёт небольшую ошибку сцены, но выбранные фрагменты после совмещения находятся далеко друг от друга.','',
        '![Правильная локализация опорной крыши для 0929](0929_reference_roof/correspondence_check.png)','',
        '![Проверка автоматически предложенной пары для 0708](0708/correspondence_check.png)','',
        '## Проверка соответствий','',
        'После выбора итоговой матрицы измерено расстояние всех точек исходного найденного фрагмента до выбранного опорного фрагмента. Диагностический фильтр требует не менее 80% точек ближе 1 м. Это явный локальный геометрический критерий, не оценка точности семантического класса. Он не участвует в подборе матрицы и не скрывает неудачные строки.','',
        '| Дата | Режим | Медиана / P90 пары, м | Доля ближе 1 м | Статус |',
        '|---|---|---:|---:|---|']
    for c in checks:
        mode='опорная крыша' if c['folder'].endswith('_reference_roof') else 'автоматические кандидаты'
        lines.append(f"| {c['date']} | {mode} | {c['forward_median_m']:.3f} / {c['forward_p90_m']:.3f} | {c['forward_fraction_within_1m']:.1%} | {c['review_status']} |")
    lines+=['','`candidate_for_review` означает кандидата для проверки, не принятую геопривязку. `object_pair_not_confirmed` — геометрия сцены прошла грубый фильтр, но объектная пара не подтверждена. `rejected_training_alignment` — отклонено по обучающей медиане > 0,5 м или медианному различию RGB > 0,35, как в прежнем пайплайне.','',
        '## Как устроен эксперимент','',
        '1. Каждая вторая запись PLY с opacity logit > 0 и конечными координатами. Применены прежние пространственные границы выборки. Все плитки с floor(x/0,4) + floor(y/0,4) + floor(z/0,4) ≡ 0 mod 4 исключены из поиска объектов и численной подгонки. Исходные индексы сохранены в `training_ids.npz`.','2. Цвет взят из DC-компоненты гауссиана. Светлые малонасыщенные точки нормализуются по размеру сцены. RANSAC выделяет плоскости, DBSCAN — связные фрагменты. Для кандидатов проверяются высота относительно предполагаемой земли, близость нормали к вертикали, прямоугольность и пропорции. Это эвристики, которые могут путать крыши, навесы и другие поверхности.','3. В режиме опорной крыши используется ранее проверенный полигон `roof_shell` из `results/las_rigid/regions.json`; внутри него в LAS выделяется доминирующая светлая плоскость. Эта область задана заранее и является дополнительным входом алгоритма.','4. Для пар кандидатов перебираются четыре поворота в плоскости. Масштаб сначала оценивается по размерам целых сцен. Начальные гипотезы ранжируются по геометрии только на обучающих точках. До четырёх лучших проходят одинаковый грубый ICP; одна лучшая по обучающей медиане — одинаковое уточнение масштаба, поворота и переноса по плоскостям.','5. Контрольный вариант начинает с четырёх PCA-ориентаций и использует те же плотные обучающие данные, пороги, число итераций и правило выбора перед тонким уточнением. Это конкретный вариант с одинаковым бюджетом уточнения, не сравнение со всеми возможными реализациями ICP.','6. После фиксации матриц проверены прежние отложенные записи: каждая десятая исходная запись с прежними фильтрами. Для сравнения фиксирован одинаковый набор внутри покрытия рабочей привязки. Неперекрывающиеся точки не скрываются: полный набор, RMS и процентили записаны в CSV/NPZ.','',
        '## Разработка, перенос и ограничения','',
        '0929 — сцена разработки, на которой уточнялись правила. Сначала широкие правила выбрали участок стройплощадки вместо крыши; более строгий порог прямоугольности 0,80 пропустил крышу в LAS. Итоговый порог 0,75 и прочие правила сохранены в аудитах. Промежуточные прогоны оставлены в `development_broad_rules`, `overstrict_rules_0929`, `overstrict_rules_0708` и `exploration`. Их лучшие числа не подставляются в итоговую таблицу.','',
        'Для остальных дат правила не перенастраивались. Однако это один и тот же объект и площадка, а опорная крыша ранее осматривалась на всех датах; такой перенос не доказывает работу на новых строительных объектах. Геометрический контроль 0929 не является слепым внешним тестом, поскольку правила разрабатывались на этой сцене.','',
        'RANSAC и прореживание Open3D могут менять выделенные фрагменты даже при фиксированном seed в зависимости от выполнения. Сохранены фактические кандидаты, их точки, гипотезы, выбранные матрицы и исходные хеши. Таблица описывает эти сохранённые прогоны; повторный расчёт не гарантирует побитовое совпадение.','',
        'Один объект с почти плоской поверхностью слабо определяет полный поворот и сдвиг; неполное покрытие изменяет контур и центр. Дальнейшая работа — проверять несколько согласованных объектных соответствий и использовать более надёжные признаки класса/контекста. Нельзя объявлять семантическую часть завершённой на основании одной удачной крыши. Направление «отдельная фотография → положение камеры в модели» остаётся следующим отдельным экспериментом.','',
        '## Результаты и запуск','',
        '[Общее сравнение](comparison.csv) · [проверки объектных пар](correspondence_checks.csv). В папках дат: `audit.json`, `comparison.csv`, `object_proposals.json`, найденные фрагменты NPZ, отложенные расстояния и изображения.','',
        '```sh','OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/align_objects.py 0929',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/align_objects.py 0929 --reference-roof',
        '.venv/bin/python scripts/report_object_alignment.py',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s scripts -p "test_*.py" -v','```','',
        'Четыре новых теста проверяют размеры прямоугольника при повороте, отбрасывание земли и чрезмерно высоких поверхностей, восстановимость известного преобразования среди гипотез и отсутствие выдуманной матрицы при отсутствии объектов. Полный набор: 23 теста.','',
        '[Контроль сохранённых результатов](validation.json): повторно рассчитаны расстояния для всех 18 итоговых матриц шести завершившихся прогонов; проверены индексы обучающих записей девяти попыток. Утечек контрольных плиток нет. Хеши 14 исходных PLY/LAS и 21 файла рабочей привязки совпадают с хешами до эксперимента. Это проверка вычислений и сохранности данных, а не подтверждение качества всех найденных привязок.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines));print(OUT/'REPORT.md')


if __name__=='__main__':main()
