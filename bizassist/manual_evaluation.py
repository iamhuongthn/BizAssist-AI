"""Fixed, hand-authored evaluation questions with independently computed reference values.

No question is synthesized from assistant output. Gold numbers are calculated from
original raw CSV/XLSX, not from assistant analytics helpers or model predictions.
The evaluator does not modify questions or gold labels after seeing answers.
"""
from pathlib import Path
import re
import time
import pandas as pd
from .generate import BASE

EVAL_DIR = BASE / 'evaluation'
RESULT_DIR = BASE / 'reports' / 'results'


def _money_in_answer(answer, label, expected):
    # Require the exact labelled monetary field; not a coincidental number elsewhere.
    pattern = re.escape(label) + r'\s*\$\s*([\d,]+\.\d{2})'
    match = re.search(pattern, answer, flags=re.I)
    return bool(match and abs(float(match.group(1).replace(',', '')) - expected) <= .011)


def _independent_real_source(path):
    raw = pd.read_excel(path, engine='openpyxl')
    raw['transaction_date'] = pd.to_datetime(raw['transaction_date'], errors='raise')
    raw['reference_revenue'] = (raw['transaction_qty'] * raw['unit_price']).round(2)
    return raw


def _real_period(raw, typ, period, store):
    d = raw
    if typ == 'month':
        d = d[d.transaction_date.dt.strftime('%Y-%m') == period]
    elif typ == 'day':
        d = d[d.transaction_date.dt.strftime('%Y-%m-%d') == period]
    elif typ == 'range':
        a,b=period.split('|')
        d = d[(d.transaction_date.dt.strftime('%Y-%m-%d') >= a) & (d.transaction_date.dt.strftime('%Y-%m-%d') <= b)]
    if store != 'All stores':
        d = d[d.store_location == store]
    return d


def _independent_synthetic_source():
    orders = pd.read_csv(BASE/'data/raw/orders.csv', parse_dates=['order_date'])
    items = pd.read_csv(BASE/'data/raw/order_items.csv')
    products = pd.read_csv(BASE/'data/raw/products.csv')
    eligible = orders.loc[orders.status.eq('completed')].copy()
    eligible['date'] = eligible.order_date.dt.strftime('%Y-%m-%d')
    lines = items.merge(eligible[['order_id','date']],on='order_id',how='inner',validate='many_to_one')
    lines = lines.merge(products[['product_id','product_name']],on='product_id',how='left',validate='many_to_one')
    lines['amount'] = lines.quantity * lines.unit_price
    return eligible,lines


def _independent_daily(orders, lines, date):
    d=orders[orders.date == date]
    l=lines[lines.date == date]
    gross=round(float(l.amount.sum()),2)
    discounts=round(float(d.discount_total.sum()),2)
    refunds=round(float(d.refund_total.sum()),2)
    net=round(gross-discounts-refunds,2)
    return dict(gross=gross,discounts=discounts,refunds=refunds,net=net,count=len(d),aov=round(net/len(d),2) if len(d) else None)


def _rank_reference(df, metric, real=False):
    if real:
        g=df.groupby(['product_id','product_detail'],as_index=False).agg(units=('transaction_qty','sum'),revenue=('reference_revenue','sum'))
    else:
        g=df.groupby(['product_id','product_name'],as_index=False).agg(units=('quantity','sum'),revenue=('amount','sum'))
    key='revenue' if metric=='revenue' else 'units'
    return g.sort_values([key,'product_id'],ascending=[False,True]).head(5)


def _rank_in_answer(answer, reference, metric, real=False):
    # Verify all five IDs in order and values, using the assistant's tabular lines.
    # Pandas to_string prints one row per product; robust to spaces in names.
    lines=answer.splitlines()
    positions=[]
    for _,r in reference.iterrows():
        pid=int(r.product_id)
        candidate=[(i,line) for i,line in enumerate(lines) if re.match(r'^\s*'+str(pid)+r'\s+',line)]
        if not candidate:return False
        i,line=candidate[0]
        tokens=line.split()
        try:
            # Synthetic: id name units gross_revenue; Real: id detail units gross_revenue
            actual_units=int(tokens[-2]);actual_rev=float(tokens[-1])
        except (ValueError,IndexError):return False
        if actual_units!=int(r.units) or abs(actual_rev-float(r.revenue))>.011:return False
        positions.append(i)
    return positions==sorted(positions)


def evaluate_manual(unified_app, real_xlsx=None, output_dir=RESULT_DIR):
    """Evaluate 15 fixed questions per dataset. Returns (summary, case_results)."""
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    real_xlsx=Path(real_xlsx) if real_xlsx else BASE/'data/real/raw/Coffee Shop Sales.xlsx'
    raw_real=_independent_real_source(real_xlsx)
    raw_orders,raw_lines=_independent_synthetic_source()
    kb=pd.read_csv(BASE/'data/raw/knowledge_base.csv').set_index('doc_id')
    records=[]
    real_cases=pd.read_csv(EVAL_DIR/'real_coffee_shop_manual_test.csv',keep_default_na=False)
    synth_cases=pd.read_csv(EVAL_DIR/'synthetic_manual_test.csv',keep_default_na=False)
    assert len(real_cases)==15 and len(synth_cases)==15
    for _,c in real_cases.iterrows():
        t=time.perf_counter()
        answer=unified_app.answer(c.question,'Real Coffee Shop (Kaggle)',c.store)
        elapsed=time.perf_counter()-t
        d=_real_period(raw_real,c.period_type,c.period,c.store)
        expected='';correct=False
        if c.kind=='revenue':
            expected=round(float(d.reference_revenue.sum()),2)
            correct=_money_in_answer(answer,'Gross transaction revenue:',expected)
        elif c.kind=='ranking':
            metric='revenue' if 'by revenue' in c.question.lower() else 'units'
            ref=_rank_reference(d,metric,real=True)
            expected='; '.join(f'{int(x.product_id)}:{int(x.units)}:{float(x.revenue):.2f}' for _,x in ref.iterrows())
            correct=_rank_in_answer(answer,ref,metric,real=True)
        else:
            expected='Explicitly declines unsupported financial field'
            correct=('unavailable' in answer.lower() and 'net sales' in answer.lower())
        records.append(dict(dataset='real',id=c.id,kind=c.kind,question=c.question,store=c.store,
                            expected=str(expected),actual_answer=answer,passed=bool(correct),
                            latency_seconds=elapsed,reference_method='Independent raw XLSX aggregation / explicit source-schema limitation'))
    for _,c in synth_cases.iterrows():
        t=time.perf_counter()
        answer=unified_app.answer(c.question,'Synthetic POS')
        elapsed=time.perf_counter()-t
        expected='';correct=False
        if c.kind=='numeric':
            ref=_independent_daily(raw_orders,raw_lines,c.start_date)
            expected=str(ref)
            correct=all([_money_in_answer(answer,'Gross sales:',ref['gross']),
                _money_in_answer(answer,'Discounts:',ref['discounts']),
                _money_in_answer(answer,'Refunds:',ref['refunds']),
                _money_in_answer(answer,'Net sales:',ref['net']),
                f'Eligible completed orders: {ref["count"]}' in answer,
                (('AOV: N/A' in answer) if ref['aov'] is None else _money_in_answer(answer,'AOV:',ref['aov']))])
        elif c.kind=='ranking':
            d=raw_lines[(raw_lines.date>=c.start_date)&(raw_lines.date<=c.end_date_or_gold_doc_id)]
            metric='revenue' if 'by revenue' in c.question.lower() else 'units'
            ref=_rank_reference(d,metric)
            expected='; '.join(f'{int(x.product_id)}:{int(x.units)}:{float(x.revenue):.2f}' for _,x in ref.iterrows())
            correct=_rank_in_answer(answer,ref,metric)
        elif c.kind=='knowledge':
            gold=c.end_date_or_gold_doc_id
            expected=f'{gold}: {kb.loc[gold,"content"]}'
            # Exact expected citation AND independently authored reference definition.
            correct=(f'Source: {gold}' in answer and str(kb.loc[gold,'content']) in answer)
        else:
            expected='Explicit unsupported-request refusal'
            correct=answer.startswith('Unsupported request:')
        records.append(dict(dataset='synthetic',id=c.id,kind=c.kind,question=c.question,store='',
                            expected=str(expected),actual_answer=answer,passed=bool(correct),
                            latency_seconds=elapsed,reference_method='Independent raw CSV joins / curated gold FAQ ID / unsupported intent'))
    results=pd.DataFrame(records)
    results.to_csv(output_dir/'manual_evaluation_results_30.csv',index=False)
    summary=results.groupby(['dataset','kind'],as_index=False).agg(cases=('passed','size'),passed=('passed','sum'),accuracy=('passed','mean'),median_latency_seconds=('latency_seconds','median'))
    totals=results.groupby('dataset',as_index=False).agg(cases=('passed','size'),passed=('passed','sum'),accuracy=('passed','mean'),median_latency_seconds=('latency_seconds','median'))
    totals['kind']='ALL'
    summary=pd.concat([summary,totals[summary.columns]],ignore_index=True)
    summary.to_csv(output_dir/'manual_evaluation_summary.csv',index=False)
    return summary,results
