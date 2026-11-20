from pathlib import Path
import pandas as pd
import numpy as np
from .generate import BASE, RAW

PROCESSED=BASE/'data'/'processed'; REJECTED=BASE/'data'/'rejected'

def quality(df):
    return {'rows':int(len(df)), 'missing_cells':int(df.isna().sum().sum()), 'duplicate_rows':int(df.duplicated().sum())}

def process(raw_dir=None):
    raw_dir=Path(raw_dir) if raw_dir else RAW
    PROCESSED.mkdir(parents=True,exist_ok=True); REJECTED.mkdir(parents=True,exist_ok=True)
    o=pd.read_csv(raw_dir/'orders.csv'); i=pd.read_csv(raw_dir/'order_items.csv'); p=pd.read_csv(raw_dir/'products.csv')
    before={k:quality(v) for k,v in [('orders',o),('items',i),('products',p)]}
    required={'orders':['order_id','order_date','status','discount_total','refund_total'],
              'items':['item_id','order_id','product_id','quantity','unit_price'],
              'products':['product_id','product_name','category']}
    for name,df in [('orders',o),('items',i),('products',p)]:
        missing=set(required[name])-set(df.columns)
        if missing: raise ValueError(f'{name} missing columns: {missing}')
    o['order_date']=pd.to_datetime(o.order_date,errors='coerce')
    for df, cols in [(o,['order_id','discount_total','refund_total']), (i,['item_id','order_id','product_id','quantity','unit_price']), (p,['product_id'])]:
        for col in cols: df[col]=pd.to_numeric(df[col],errors='coerce')
    o_bad=o.order_id.isna()|o.order_id.duplicated(keep='first')|o.order_date.isna()|~o.status.isin(['completed','cancelled','failed'])|o.discount_total.isna()|o.refund_total.isna()|(o.discount_total<0)|(o.refund_total<0)
    p_bad=p.product_id.isna()|p.product_id.duplicated(keep='first')|p.product_name.isna()|p.category.isna()
    good_o=o.loc[~o_bad].copy(); good_p=p.loc[~p_bad].copy()
    i_bad=i.item_id.isna()|i.item_id.duplicated(keep='first')|i.order_id.isna()|i.product_id.isna()|i.quantity.isna()|i.unit_price.isna()|(i.quantity<=0)|(i.unit_price<0)|~i.order_id.isin(good_o.order_id)|~i.product_id.isin(good_p.product_id)
    good_i=i.loc[~i_bad].copy()
    # Quarantine entire orders with any rejected items or inconsistent financial adjustments.
    invalid_order_ids=set(i.loc[i_bad & i.order_id.notna(),'order_id'])
    gross=good_i.assign(line_gross=lambda d:d.quantity*d.unit_price).groupby('order_id').line_gross.sum()
    total=good_o.order_id.map(gross).fillna(0)
    financial_bad=(good_o.discount_total+good_o.refund_total>total+0.000001) | ((good_o.status!='completed') & ((good_o.discount_total!=0)|(good_o.refund_total!=0))) | good_o.order_id.isin(invalid_order_ids) | (total<=0)
    rejected_o=pd.concat([o.loc[o_bad].assign(rejection_reason='schema/ID/status/date/value'),good_o.loc[financial_bad].assign(rejection_reason='invalid item or financial reconciliation')])
    good_o=good_o.loc[~financial_bad].copy()
    good_i=good_i[good_i.order_id.isin(good_o.order_id)].copy()
    rejected_o.to_csv(REJECTED/'orders_rejected.csv',index=False)
    i.loc[i_bad].to_csv(REJECTED/'items_rejected.csv',index=False)
    p.loc[p_bad].to_csv(REJECTED/'products_rejected.csv',index=False)
    for name,df in [('orders',good_o),('order_items',good_i),('products',good_p)]: df.to_csv(PROCESSED/f'{name}.csv',index=False)
    faq=pd.read_csv(raw_dir/'knowledge_base.csv').dropna(subset=['doc_id','content','source','updated_at']).copy()
    faq['content']=faq.content.astype(str).str.normalize('NFKC').str.replace(r'\s+',' ',regex=True).str.strip()
    faq=faq[faq.content.str.len()>0].drop_duplicates(subset=['doc_id'])
    faq.to_csv(PROCESSED/'knowledge_base.csv',index=False)
    after={k:quality(v) for k,v in [('orders',good_o),('items',good_i),('products',good_p)]}
    report=pd.DataFrame([{'table':k,**{f'before_{a}':v for a,v in before[k].items()},**{f'after_{a}':v for a,v in after[k].items()},'rejected_rows':before[k]['rows']-after[k]['rows']} for k in before])
    report.to_csv(BASE/'reports'/'results'/'quality_metrics.csv',index=False)
    return good_o,good_i,good_p,faq,report
