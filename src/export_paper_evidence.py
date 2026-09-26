"""Recompute paper assets from case-level results, without manuscript targets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == 'code' else HERE
CODE_ROOT = ROOT / 'code' if (ROOT / 'code').exists() else ROOT
sys.path.insert(0, str(CODE_ROOT / 'directional_validation'))
from run_directional import SELECTORS, calibration_offsets, evidence_decisions

FAMILIES = ['monthly industry', 'monthly size--B/M', 'monthly size/profit/invest',
            'monthly two-characteristic', 'daily size characteristic', 'daily industry/factor']
LABELS = ['Monthly industry', 'Monthly size--B/M', 'Monthly size/profit/invest',
          'Monthly two-characteristic', 'Daily size characteristic', 'Daily industry/factor']


def match_acceptance(natural):
    """Retrospective batch budget; rank by evidence only, preserve rejections."""
    result = natural.copy()
    result['natural_decision'] = result.decision
    result['budget_abstain'] = False
    for seed, g in result.groupby('seed', sort=True):
        available = g.groupby('selector').decision.apply(lambda s: s.eq('ACCEPT').sum())
        groups = int(g.groupby('selector').size().iloc[0])
        k = min(int(np.floor(.75 * groups)), int(available.min()))
        for selector, sub in g.groupby('selector', sort=True):
            # Stable input order comes from lexicographic task/behavior grouping.
            ordered = sub[sub.decision.eq('ACCEPT')].sort_values('evidence_score', ascending=False, kind='stable')
            remove = ordered.index[k:]
            result.loc[remove, 'decision'] = 'ABSTAIN'
            result.loc[remove, ['false_safe', 'unsafe_accept']] = 0
            result.loc[remove, 'budget_abstain'] = True
            result.loc[sub.index, 'common_budget'] = k
    return result


def summary(decisions):
    rows = []
    for sel in SELECTORS:
        g = decisions[decisions.selector.eq(sel)]
        a = int(g.decision.eq('ACCEPT').sum())
        rows.append(dict(selector=sel, settings=len(g), accepted=a,
                         rejected=int(g.decision.eq('REJECT').sum()),
                         abstained=int(g.decision.eq('ABSTAIN').sum()),
                         budget_abstained=int(g.get('budget_abstain', pd.Series(False, index=g.index)).sum()),
                         false_safe_count=int(g.false_safe.sum()), unsafe_accept_count=int(g.unsafe_accept.sum()),
                         false_safe_pct=100*g.false_safe.sum()/a if a else np.nan,
                         unsafe_accept_pct=100*g.unsafe_accept.sum()/a if a else np.nan))
    return pd.DataFrame(rows)


def cluster_ci(decisions):
    tasks = sorted(decisions.task.unique())
    indices = np.random.default_rng(1729).integers(0, len(tasks), (2000, len(tasks)))
    records = []
    for metric in ['false_safe', 'unsafe_accept']:
        values = {}
        for sel in SELECTORS:
            d = decisions[decisions.selector.eq(sel)].copy()
            d['accepted'] = d.decision.eq('ACCEPT').astype(int)
            x = d.groupby('task')[[metric, 'accepted']].sum().reindex(tasks).to_numpy()
            draws = x[indices].sum(axis=1)
            values[sel] = (100*x[:,0].sum()/x[:,1].sum(), 100*draws[:,0]/draws[:,1])
        for baseline in SELECTORS[1:]:
            difference = values['SCROPE++'][1] - values[baseline][1]
            lo, hi = np.quantile(difference, [.025, .975])
            records.append(dict(metric=metric, baseline=baseline,
                                difference_pp=values['SCROPE++'][0]-values[baseline][0],
                                low_pp=lo, high_pp=hi, draws=2000, cluster='task'))
    return pd.DataFrame(records)


def latex_table(path, caption, label, columns, rows, wide=False):
    env = 'table*' if wide else 'table'
    alignment = 'l' + 'r' * (len(columns)-1)
    content = [r'\begin{'+env+'}[t]', r'\centering\scriptsize', r'\setlength{\tabcolsep}{3pt}',
               r'\caption{'+caption+'}', r'\label{'+label+'}', r'\begin{tabular}{'+alignment+'}',
               r'\toprule', ' & '.join(columns)+r' \\', r'\midrule']
    content += [' & '.join(map(str, row))+r' \\' for row in rows]
    content += [r'\bottomrule', r'\end{tabular}', r'\end{'+env+'}']
    path.write_text('\n'.join(content)+'\n', encoding='utf-8')


def figures(paper, coverage, selectors, cells):
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, ax = plt.subplots(figsize=(5.8,2.7), layout='constrained')
    x = np.arange(5)
    for scheme, color in [('iid','#566C8C'),('block','#22877A')]:
        d = coverage[coverage.scheme.eq(scheme)].sort_values('ess_bin')
        ax.plot(x, 100*d.nominal_coverage, 'o-', label=scheme.upper()+' bootstrap', color=color, lw=1.3, ms=4)
        ax.plot(x, 100*d.calibrated_coverage, 's--', label=scheme.upper()+' + calibration', color=color, lw=1.2, ms=3)
    ax.axhline(95, color='#8C8C8C', lw=.8, ls=':')
    ax.set(xticks=x, xticklabels=['[0,2)','[2,5)','[5,10)','[10,50)','50+'], xlabel='Effective sample size', ylabel='Coverage (%)', ylim=(0,105))
    ax.legend(loc='lower right', fontsize=7, frameon=False)
    fig.savefig(paper/'coverage_current.pdf', bbox_inches='tight')
    fig.savefig(paper/'coverage_by_ess_clean.png', dpi=220, bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.8,2.7), layout='constrained')
    x = np.arange(4)
    for dx, col, label, color in [(-.18,'false_safe_pct','Strict false-safe','#59708C'),(.18,'unsafe_accept_pct','Unsafe acceptance','#288879')]:
        bars=ax.bar(x+dx, selectors[col], width=.34, label=label, color=color)
        ax.bar_label(bars, fmt='%.2f', fontsize=7, padding=2)
    ax.set(xticks=x,xticklabels=['SCROPE++','Support-only','LCB-only','Risk-only'],ylabel='Among ACCEPT decisions (%)',ylim=(0,80))
    ax.legend(frameon=False, fontsize=7,loc='upper right')
    fig.savefig(paper/'selectors_current.pdf', bbox_inches='tight')
    fig.savefig(paper/'matched_acceptance_rates.png', dpi=220, bbox_inches='tight')
    plt.close(fig)
    # Compact single-row pipeline figure: the camera-ready paper is limited to
    # eight pages, so keep the workflow legible while minimizing vertical use.
    fig,ax=plt.subplots(figsize=(7.16,0.92))
    ax.set(xlim=(0,11.55),ylim=(0,1)); ax.axis('off')
    titles=['Public panels','Reward matrix','Logged feedback','OPE diagnostics','SCROPE++']
    subtitles=['15 tasks',f'{cells/1e6:.3f}M cells\n20 lagged features','8 behaviors\n3 seeds','DR intervals\nweighted CVaR95','Return LCB\nrisk + support']
    colors=['#EEF1F4','#EDF3FA','#FCF4DE','#EAF5F2','#F9EFEC']
    for i,(title,sub,color) in enumerate(zip(titles,subtitles,colors)):
        left=i*2.30
        ax.add_patch(FancyBboxPatch((left,.20),1.95,.58,boxstyle='round,pad=0.02,rounding_size=0.03',fc=color,ec='#5B6775',lw=.7))
        ax.text(left+.975,.61,title,ha='center',va='center',fontsize=7.5,weight='bold')
        ax.text(left+.975,.35,sub,ha='center',va='center',fontsize=5.2,linespacing=1.05)
        if i<4: ax.annotate('',xy=(left+2.25,.49),xytext=(left+2.01,.49),arrowprops=dict(arrowstyle='->',lw=.7,mutation_scale=4,color='#364152'))
    fig.savefig(paper/'pipeline_current.pdf',bbox_inches='tight',pad_inches=.02)
    fig.savefig(paper/'pipeline_clean.png', dpi=220, bbox_inches='tight', pad_inches=.02)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--paper',type=Path,required=True)
    args=p.parse_args(); run=args.run; paper=args.paper
    tables=paper/'generated_tables'; audit=paper/'evidence'
    tables.mkdir(parents=True,exist_ok=True); audit.mkdir(parents=True,exist_ok=True)
    cases=pd.read_csv(run/'all_cases.csv')
    cal=cases[cases.split.eq('calibration')]; test=cases[cases.split.eq('test')].copy()
    offsets=calibration_offsets(cal)
    look=offsets[offsets.kind.eq('return')].set_index(['scheme','ess_bin']).offset
    test['offset']=[look.loc[(s,b)] for s,b in zip(test.scheme,test.ess_bin)]
    test['nominal_covered']=(test.return_lcb<=test.truth_return)&(test.truth_return<=test.return_ucb)
    test['calibrated_covered']=(test.return_lcb-test.offset<=test.truth_return)&(test.truth_return<=test.return_ucb+test.offset)
    test['width']=test.return_ucb-test.return_lcb
    test['cal_width']=test.width+2*test.offset
    test['under']=(test.risk_hat<test.truth_cvar-1e-12).where(test.risk_hat.notna())
    test['risk_error']=abs(test.risk_hat-test.truth_cvar)
    coverage=test.groupby(['scheme','ess_bin']).agg(n=('task','size'),nominal_coverage=('nominal_covered','mean'),calibrated_coverage=('calibrated_covered','mean'),width=('width','mean'),cal_width=('cal_width','mean')).reset_index()
    risk=test[test.scheme.eq('block')].groupby('family').agg(n=('task','size'),valid=('risk_hat','count'),under=('under','mean'),mae=('risk_error','mean'),median_ess=('ESS','median')).reindex(FAMILIES).reset_index()
    natural=[]
    for keys,g in test[test.scheme.eq('block')].groupby(['task','family','behavior','seed'],sort=True):
        for row in evidence_decisions(g,float(g.baseline_tau.iloc[0]),'risk_ucb'):
            row.update(zip(['task','family','behavior','seed'],keys)); natural.append(row)
    natural=pd.DataFrame(natural); matched=match_acceptance(natural)
    selectors=summary(matched); natural_summary=summary(natural); ci=cluster_ci(matched)
    inventory=pd.read_json(run/'task_inventory.json')
    families=cases[['task','family']].drop_duplicates()
    inventory=inventory.merge(families,on='task',validate='one_to_one')
    for name,df in [('coverage',coverage),('risk_diagnostics',risk),('natural_decisions',natural),('matched_decisions',matched),('matched_summary',selectors),('natural_summary',natural_summary),('paired_intervals',ci),('task_inventory',inventory),('calibration_offsets',offsets)]:
        df.to_csv(audit/(name+'.csv'),index=False)
    def span(x):
        lo,hi=int(x.min()),int(x.max())
        return str(lo) if lo==hi else f'{lo}--{hi}'
    rows=[]
    for fam,label in zip(FAMILIES,LABELS):
        g=inventory[inventory.family.eq(fam)]
        rows.append([label,len(g),span(g.periods),span(g.actions),20,span(g.train_n),span(g.calibration_n),span(g.test_n),f'{g.cells.sum():,}'])
    rows.append(['Total',15,'','','','','','',f'{inventory.cells.sum():,}'])
    latex_table(tables/'table_benchmark_summary.tex','Current benchmark after filtering and 60-date warmup. Periods and cells cover train, calibration, and test; feature count is 20 for every task.','tab:benchmark_summary',['Family','Tasks','Periods','Actions','Features','Train','Cal.','Test','Cells'],rows,True)
    cr=[]
    bins=['0--2','2--5','5--10','10--50','50+']
    for i in range(5):
        iid=coverage[(coverage.scheme=='iid')&(coverage.ess_bin==i)].iloc[0]; block=coverage[(coverage.scheme=='block')&(coverage.ess_bin==i)].iloc[0]
        cr.append([bins[i],int(block.n),f'{100*iid.nominal_coverage:.2f}',f'{100*block.nominal_coverage:.2f}',f'{100*block.calibrated_coverage:.2f}',f'{block.width:.4f}',f'{block.cal_width:.4f}'])
    latex_table(tables/'table_ope_by_support.tex','DR interval coverage by ESS. IID, Block, and Cal. are percentages; Cal. applies support-bin offsets to block intervals. Widths are in return units.','tab:ope_by_support',['ESS','$N$','IID','Block','Cal.','Width','Cal. width'],cr)
    rr=[]
    short=['Monthly industry','Monthly size--B/M','Monthly size/P/I','Monthly two-char.','Daily size-char.','Daily industry/factor']
    for label,row in zip(short,risk.itertuples()):
        rr.append([label,int(row.valid),f'{100*row.under:.2f}',f'{row.mae:.4f}',f'{row.median_ess:.2f}'])
    latex_table(tables/'table_risk_ope_by_family.tex','Test-window CVaR95 diagnostics. Underestimation (Under., \\%) and MAE exclude undefined risk estimates; median ESS uses all cases.','tab:risk_family',['Family','Valid','Under.','MAE','ESS'],rr)
    sr=[]
    for r in selectors.itertuples():
        sr.append([r.selector,r.accepted,r.rejected,r.abstained,f'{r.false_safe_pct:.2f}',f'{r.unsafe_accept_pct:.2f}'])
    latex_table(tables/'table_matched_abstention_comparison.tex','Matched-acceptance audit, 360 groups per selector. FS and UA are strict false-safe and unsafe-acceptance percentages among ACCEPT decisions. ABSTAIN includes budget withholding.','tab:reviewer_aligned',['Selector','Accept','Reject','Abstain','FS','UA'],sr)
    figures(paper,coverage,selectors,int(inventory.cells.sum()))
    proof={'input_sha256':{f:hashlib.sha256((run/f).read_bytes()).hexdigest() for f in ['all_cases.csv','task_inventory.json','manifest.json']},'case_rows':len(cases),'test_rows_per_scheme':int(test.scheme.eq('block').sum()),'risk_evaluable':int(risk.valid.sum()),'reward_cells':int(inventory.cells.sum()),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (audit/'number_trace.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
    print(selectors.to_string(index=False)); print(ci.to_string(index=False)); print(json.dumps(proof,indent=2))


if __name__=='__main__':
    main()
