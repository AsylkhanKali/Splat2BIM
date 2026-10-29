"""Report actual control accuracy, supervision and tradeoffs for seven dates."""
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from align_dates import ROOT,DATES
from align_0929 import apply
from fit_structural_refinement import OUT as FIRST
from fit_guarded_structural_refinement import OUT as GUARDED
from fit_reviewed_structural_refinement import OUT,BASE
from check_las_vertical import write_csv,file_hash

METHODS=['current_automatic','stable_parent','structural_refinement']


def read_csv(path):return list(csv.DictReader(path.open()))


def main():
    statics=[];global_rows=[];auto_controls=[];summaries=[];support=[];trials=[]
    for date in DATES:
        folder=OUT/date;fit=json.loads((folder/'fit.json').read_text());evaluation=json.loads((folder/'evaluation.json').read_text())
        statics+=read_csv(folder/'static_structures.csv');global_rows+=read_csv(folder/'comparison.csv');auto_controls+=read_csv(folder/'structural_controls.csv')
        row={'date':date,'status':fit['status'],'selected_candidate':fit['selected_candidate'],
            'strict_support_count':evaluation['selected_evidence']['strict_forward_080_spatial_support_count']}
        for method in METHODS:
            values=[];tails=[]
            for region in ['roof','bank_mid']:
                metric=next(r for r in statics if r['date']==date and r['method']==method and r['region']==region and r['scope']=='unseen_records')
                assert int(metric['count'])>=30;value=float(metric['median']);tail=float(metric['p90'])
                row[method+'_'+region+'_median_m']=value;row[method+'_'+region+'_p90_m']=tail
                row[region+'_control_records']=int(metric['count']);values.append(value);tails.append(tail)
            row[method+'_mean_region_median_m']=float(np.mean(values));row[method+'_mean_region_p90_m']=float(np.mean(tails))
            metric=next(r for r in global_rows if r['date']==date and r['method']==method and r['scope']=='fixed_baseline_coverage')
            for key in ['median','p90']:row[method+'_global_'+key+'_m']=float(metric[key])
        row['mean_region_median_improvement_m']=row['stable_parent_mean_region_median_m']-row['structural_refinement_mean_region_median_m']
        row['mean_region_median_improvement_percent']=100*row['mean_region_median_improvement_m']/row['stable_parent_mean_region_median_m']
        p=np.load(folder/'held_out_distances.npz')['source_points'];old=apply(p,np.array(fit['initial_matrix']));new=apply(p,np.array(fit['matrix_local']))
        row['pose_motion_p95_m']=float(np.quantile(np.linalg.norm(new-old,axis=1),.95));summaries.append(row)
        for pair in evaluation['selected_evidence']['strict_spatial_support']:
            support.append({'date':date,'source_id':pair['source_id'],'target_id':pair['target_id'],
                'forward_fraction':pair['fraction_within_1m'],'reverse_fraction':pair['reverse_fraction_within_1m']})
        for root,name in [(FIRST,'automatic_structures_only'),(GUARDED,'automatic_context_guarded'),(OUT,'reviewed_regions')]:
            saved=json.loads((root/date/'fit.json').read_text());stats=read_csv(root/date/'static_structures.csv')
            actual=[r for r in stats if r['method']=='structural_refinement' and r['scope']=='unseen_records']
            trials.append({'date':date,'experiment':name,'status':saved['status'],
                'mean_control_region_median_m':float(np.mean([float(r['median']) for r in actual])),
                'strict_support_count':json.loads((root/date/'evaluation.json').read_text())['selected_evidence']['strict_forward_080_spatial_support_count']})
    for filename,rows in [('summary.csv',summaries),('static_structures.csv',statics),('comparison.csv',global_rows),
        ('automatic_structural_controls.csv',auto_controls),('strict_support.csv',support),('experiments.csv',trials)]:write_csv(OUT/filename,rows)
    improved=[r['date'] for r in summaries if r['mean_region_median_improvement_m']>.001]
    regressed=[r['date'] for r in summaries if r['mean_region_median_improvement_m']<-.001]
    unchanged=[r['date'] for r in summaries if abs(r['mean_region_median_improvement_m'])<=.001]
    before=float(np.mean([r['stable_parent_mean_region_median_m'] for r in summaries]));after=float(np.mean([r['structural_refinement_mean_region_median_m'] for r in summaries]))
    tail_regressions=[r['date'] for r in summaries if r['structural_refinement_global_p90_m']>r['stable_parent_global_p90_m']+.01]
    global_median_increases=[r['structural_refinement_global_median_m']-r['stable_parent_global_median_m'] for r in summaries if r['status']=='refined']
    regional_regressions=[{'date':r['date'],'region':k,'increase_m':r['structural_refinement_'+k+'_median_m']-r['stable_parent_'+k+'_median_m']}
        for r in summaries for k in ['roof','bank_mid'] if r['structural_refinement_'+k+'_median_m']>r['stable_parent_'+k+'_median_m']+.001]
    spatial_roof=[]
    for date in DATES:
        before_row=next(r for r in statics if r['date']==date and r['method']=='stable_parent' and r['region']=='roof' and r['scope']=='spatial_holdout')
        after_row=next(r for r in statics if r['date']==date and r['method']=='structural_refinement' and r['region']=='roof' and r['scope']=='spatial_holdout')
        spatial_roof.append({'date':date,'count':int(before_row['count']),'adequate':int(before_row['count'])>=30,
            'before_m':float(before_row['median']),'after_m':float(after_row['median'])})
    summary={'refined_dates':[r['date'] for r in summaries if r['status']=='refined'],'retained_dates':[r['date'] for r in summaries if r['status']=='baseline_retained'],
        'improved_over_1mm':improved,'regressed_over_1mm':regressed,'unchanged_within_1mm':unchanged,
        'mean_of_date_region_medians_before_m':before,'mean_of_date_region_medians_after_m':after,'relative_improvement_percent':100*(before-after)/before,
        'global_p90_regressions_over_1cm':tail_regressions,'individual_region_median_regressions_over_1mm':regional_regressions,
        'global_median_increase_range_m':[min(global_median_increases),max(global_median_increases)],
        'spatial_roof_checks':spatial_roof,
        'strict_pass_dates':[r['date'] for r in summaries if r['strict_support_count']>=3],
        'fit_sha256':{d:file_hash(OUT/d/'fit.json') for d in DATES},'evaluation_sha256':{d:file_hash(OUT/d/'evaluation.json') for d in DATES},
        'source_sha256':{'report_structural_refinement.py':file_hash(ROOT/'scripts/report_structural_refinement.py')}}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True);x=np.arange(7)
    for ax,key,title in zip(axes.flat,['roof_median_m','bank_mid_median_m','mean_region_median_m','global_p90_m'],
        ['Крыша: медиана на неиспользованных записях','Берег: медиана на неиспользованных записях','Среднее двух медиан: равный вес областей','Вся сцена: P90 на пространственном контроле']):
        ax.bar(x-.18,[r['stable_parent_'+key] for r in summaries],.34,label='До уточнения',color='#748397')
        ax.bar(x+.18,[r['structural_refinement_'+key] for r in summaries],.34,label='После уточнения',color='#208473')
        ax.set(xticks=x,xticklabels=DATES,title=title,ylabel='Метры; меньше — лучше');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(fontsize=8)
    fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
    headline=f'Среднее по семи датам и двум областям: {before:.3f} → {after:.3f} м ({100*(before-after)/before:.1f}% снижения).'
    table=['| Дата | Крыша: до → после, м | Берег: до → после, м | Средняя медиана: до → после, м | Изменение | Строгих опор |',
        '|---|---:|---:|---:|---|---:|'];htmlrows=[]
    for r in summaries:
        values=[r['date']]+[f'{r["stable_parent_"+k]:.3f} → {r["structural_refinement_"+k]:.3f}' for k in ['roof_median_m','bank_mid_median_m','mean_region_median_m']]
        values+=[f'{r["mean_region_median_improvement_percent"]:+.1f}% снижения' if r['status']=='refined' else 'Прежняя поза',str(r['strict_support_count'])]
        table.append('| '+' | '.join(values)+' |');htmlrows.append('<tr>'+''.join('<td>'+v+'</td>' for v in values)+'</tr>')
    limits=[
        'Это уточнение использует существующую ручную разметку двух неподвижных областей — крыши и берега. Оно не является новым полностью автоматическим детектором конструкций.',
        'В подгонке участвуют только чётные исходные записи 3DGS вне прежних отложенных пространственных ячеек. Внутри обучающих данных выделена отдельная блочная выборка для выбора поправки. Показанные основные ошибки вычислены на прежних записях 5, 15, 25…; они не участвуют в подгонке или выборе.',
        'Разметка областей теперь служит обучающей подсказкой. Поэтому проверка на других записях этих же областей слабее проверки на совершенно новых конструкциях. Восстановленные центры гауссианов также не являются независимыми геодезическими измерениями.',
        'Сильный пространственный контроль сохранён отдельно. В контрольной крыше достаточно точек только на 0331, 0504, 0609, 0723 и 0831; на берегу во всех датах их меньше 30. Недостаточные выборки не засчитываются как доказательство улучшения.',
        'Масштаб зафиксирован. Проверены подгонки по плоскостям, точкам и их комбинации, затем одинаковые доли поправки. Выбор использует только обучающую и внутреннюю блочную выборки. Сдвиг всей сцены ограничен 0,5 м на фиксированных контрольных вершинах её объёма.',
        'Прежние строгие условия сохранены: ≥3 разных опор; прямое перекрытие ≥80%, обратное ≥50% в пределах 1 м; расстояние между опорами ≥10 м, вторая ось ≥2 м; центр ≤5 м, нормали ≤20°, размеры 0,5–2. Банк пар фиксирован до уточнения, новые идентичности после подгонки не добавляются.',
        'Первый автоматический опыт улучшал выбранные опоры, но ухудшал берег, например на 0504 примерно на 9 см. Вариант с контекстом сцены и запретом роста обучающих медианы/P90 сохранил прежние позы всех семи дат. Оба опыта сохранены; experiments.csv позволяет сравнить их с уточнением по размеченным областям.',
        'Метод разрабатывался на этой площадке с учётом предыдущих контрольных результатов. Это экспериментальное сравнение, а не независимый тест переноса. Полные рабочие сцены и BIM в просмотрщике не перезаписаны; новые локальные и глобальные матрицы сохранены в папках дат.'
    ]
    tradeoff=f'Медиана всей сцены на изменённых датах выросла на {1000*min(global_median_increases):.1f}–{1000*max(global_median_increases):.1f} мм. Даты с ростом P90 всей сцены более 1 см: '+(', '.join(tail_regressions) or 'нет')+'. Поэтому новые матрицы сохранены отдельно и не заменяют рабочие экспорты автоматически.'
    regressions='Области с ростом медианы более 1 мм: '+('; '.join(f'{r["date"]} {r["region"]}: +{1000*r["increase_m"]:.1f} мм' for r in regional_regressions) or 'нет')+'.'
    spatial_text='На более строгом пространственном контроле крыши медиана снизилась на 0331, 0504, 0723 и 0831; на 0609 выросла на 3,3 мм. На 0708 и 0929 эта выборка недостаточна. На берегу пространственных контрольных точек недостаточно во всех датах.'
    lines=['# Уточнение точности по размеченным неподвижным конструкциям','',headline,'',
        f'Строгий допуск: {len(summary["strict_pass_dates"])}/7. Приняты поправки: '+', '.join(summary['refined_dates'])+'. Прежняя поза: '+(', '.join(summary['retained_dates']) or 'нет')+'.','',*table,'',tradeoff,'',regressions,'',
        '![Сравнение](comparison.png)','',spatial_text,'', '## Условия и ограничения','',*sum(([p,''] for p in limits),[]),
        '## Проверка всей сцены','', '| Дата | Медиана до → после, м | P90 до → после, м |','|---|---:|---:|']
    for r in summaries:lines.append('| '+r['date']+' | '+' | '.join(f'{r["stable_parent_global_"+k+"_m"]:.3f} → {r["structural_refinement_global_"+k+"_m"]:.3f}' for k in ['median','p90'])+' |')
    lines+=['','Сравнение с исходным автоматическим методом и последним вариантом устойчивых поверхностей находится в summary.csv. Сырые ошибки двух видов контрольных выборок — static_structures.csv; дополнительные автоматические поверхности — automatic_structural_controls.csv. Все кандидаты, включая непринятые, сохранены в <дата>/fit.json и candidate_comparison.csv.','',
        '## Воспроизведение','', '```sh','.venv/bin/python scripts/fit_structural_refinement.py','.venv/bin/python scripts/fit_guarded_structural_refinement.py',
        '.venv/bin/python scripts/fit_reviewed_structural_refinement.py',
        ".venv/bin/python -c \"import sys;sys.path.insert(0,'scripts');from evaluate_structural_refinement import main;from fit_reviewed_structural_refinement import OUT;main(OUT)\"",
        '.venv/bin/python scripts/verify_structural_refinement.py','.venv/bin/python scripts/report_structural_refinement.py','```','',
        'Численная перепроверка сохранена в validation.json. Первые два опыта имеют собственные каталоги results/structural_refinement и results/guarded_structural_refinement. Все исходные данные и предыдущие матрицы сохранены.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    page=['<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Точность по неподвижным конструкциям</title>',
        '<style>body{font:16px/1.6 system-ui,sans-serif;color:#17202a;max-width:1300px;margin:24px auto;padding:0 20px}p{max-width:105ch}img{width:100%;height:auto}a{color:#175794}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:9px;text-align:left;border-bottom:1px solid #ccd2d8}.table{overflow:auto}summary{cursor:pointer;font-weight:650;padding:14px 0}h1{font-size:clamp(24px,3vw,36px)}a:focus-visible,summary:focus-visible{outline:2px solid #175794}</style>',
        '<a href="/viewer/">← 3D-просмотрщик</a><h1>Точность по неподвижным конструкциям</h1>',
        '<p><strong>'+headline+'</strong> Строгие пороги сохранены; допуск '+str(len(summary['strict_pass_dates']))+'/7.</p>',
        '<p>Уточнение использует заранее размеченные области крыши и берега. Ни одна контрольная запись не участвует в подгонке или выборе поправки.</p>',
        '<p><a href="REPORT.md">Методика</a> · <a href="summary.csv">Сравнение семи дат</a> · <a href="static_structures.csv">Контрольные выборки</a> · <a href="experiments.csv">Все три опыта</a> · <a href="validation.json">Проверка расчётов</a></p>',
        '<div class="table"><table><thead><tr><th>Дата</th><th>Крыша, м</th><th>Берег, м</th><th>Средняя медиана, м</th><th>Результат</th><th>Строгих опор</th></tr></thead><tbody>',*htmlrows,
        '</tbody></table></div><p>До → после. Равный вес двух областей и каждой даты; ошибки на прежних неиспользованных записях.</p>',
        '<p>'+tradeoff+' '+regressions+'</p><p>'+spatial_text+'</p><img src="comparison.png" alt="Изменение ошибок крыши, берега и всей сцены на семи датах">',
        '<details><summary>Что использовано и насколько независима проверка</summary>',*['<p>'+p+'</p>' for p in limits],'</details></html>']
    (OUT/'inspection.html').write_text('\n'.join(page)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
