"""Kaggle Coffee Shop Sales adapter: separate real-data path, no fabricated refunds/net sales.
Source: https://www.kaggle.com/datasets/keremkarayaz/coffee-shop-sales
"""
from pathlib import Path
import numpy as np
import pandas as pd
from .generate import BASE

REAL = BASE / 'data' / 'real'
SOURCE = REAL / 'raw' / 'Coffee Shop Sales.xlsx'
OUTPUT = REAL / 'processed'
FIGURES = BASE / 'reports' / 'figures'
RESULTS = BASE / 'reports' / 'results'
REQUIRED = ['transaction_id','transaction_date','transaction_time','transaction_qty','store_id',
            'store_location','product_id','unit_price','product_category','product_type','product_detail']


def load_and_prepare(path=SOURCE, save=True):
    """Load real XLSX; validate and map to BizAssist-compatible tables.
    transaction_id is unique in this particular file (one row per invoice/line).
    Never infer discounts, refunds, status, or net sales from absent source fields.
    """
    path=Path(path)
    if not path.exists(): raise FileNotFoundError(f'Missing Kaggle XLSX: {path}')
    raw=pd.read_excel(path,engine='openpyxl')
    missing=set(REQUIRED)-set(raw.columns)
    if missing: raise ValueError(f'Missing required Kaggle columns: {sorted(missing)}')
    d=raw.copy()
    # Preserve original record count and a reason for every invalid row.
    d['transaction_date']=pd.to_datetime(d.transaction_date,errors='coerce')
    d['transaction_qty']=pd.to_numeric(d.transaction_qty,errors='coerce')
    d['unit_price']=pd.to_numeric(d.unit_price,errors='coerce')
    d['transaction_id']=pd.to_numeric(d.transaction_id,errors='coerce')
    d['product_id']=pd.to_numeric(d.product_id,errors='coerce')
    d['store_id']=pd.to_numeric(d.store_id,errors='coerce')
    timepart=pd.to_timedelta(d.transaction_time.astype(str),errors='coerce')
    # datetime.time objects stringify as HH:MM:SS
    d['order_date']=d.transaction_date.dt.normalize()+timepart
    rules={
        'missing_or_duplicate_transaction_id': d.transaction_id.isna()|d.transaction_id.duplicated(keep='first'),
        'invalid_date_or_time':d.order_date.isna(),
        'invalid_quantity':d.transaction_qty.isna()|(d.transaction_qty<=0),
        'invalid_price':d.unit_price.isna()|(d.unit_price<=0),
        'missing_product_or_store_id':d.product_id.isna()|d.store_id.isna(),
        'missing_product_or_store_label':d[['store_location','product_detail','product_category']].isna().any(axis=1)
    }
    reasons=pd.DataFrame(rules,index=d.index)
    invalid=reasons.any(axis=1)
    rejected=d.loc[invalid].copy()
    rejected['rejection_reason']=[ '; '.join(reasons.columns[row.to_numpy()]) for _,row in reasons.loc[invalid].iterrows() ]
    valid=d.loc[~invalid].copy()
    valid['order_id']=valid.transaction_id.astype('int64')
    valid['item_id']=np.arange(1,len(valid)+1,dtype='int64')
    valid['product_id']=valid.product_id.astype('int64')
    valid['store_id']=valid.store_id.astype('int64')
    valid['quantity']=valid.transaction_qty.astype('int64')
    valid['gross_line_amount']=(valid.quantity*valid.unit_price).round(2)
    valid['date']=valid.order_date.dt.strftime('%Y-%m-%d')
    valid['day_of_week']=valid.order_date.dt.dayofweek
    valid['hour_of_day']=valid.order_date.dt.hour
    valid['month']=valid.order_date.dt.to_period('M').astype(str)
    valid['status']='completed_assumed'  # proxy, not a source-provided POS status
    valid['currency']='USD'  # dataset is NYC-based; source file has no explicit currency field
    # Do not add synthetic 0 refund/discount fields to real dataset.
    orders=valid[['order_id','order_date','status','store_id','store_location','date','day_of_week','hour_of_day']].copy()
    items=valid[['item_id','order_id','product_id','quantity','unit_price','gross_line_amount']].copy()
    products=(valid[['product_id','product_detail','product_category','product_type']]
              .drop_duplicates('product_id').rename(columns={'product_detail':'product_name','product_category':'category'}))
    quality=pd.DataFrame([{'dataset':'Kaggle Coffee Shop Sales','raw_rows':len(raw),'valid_rows':len(valid),
        'rejected_rows':len(rejected),'raw_missing_cells':int(raw.isna().sum().sum()),
        'valid_missing_cells':int(valid[REQUIRED].isna().sum().sum()),
        'raw_duplicate_transaction_ids':int(raw.transaction_id.duplicated().sum()),
        'unique_stores':int(valid.store_id.nunique()),'unique_products':int(valid.product_id.nunique()),
        'start_date':valid.date.min(),'end_date':valid.date.max()}])
    if save:
        for p in (OUTPUT,FIGURES,RESULTS):p.mkdir(parents=True,exist_ok=True)
        valid.to_csv(OUTPUT/'coffee_shop_validated.csv',index=False)
        rejected.to_csv(OUTPUT/'coffee_shop_rejected.csv',index=False)
        orders.to_csv(OUTPUT/'orders.csv',index=False)
        items.to_csv(OUTPUT/'order_items.csv',index=False)
        products.to_csv(OUTPUT/'products.csv',index=False)
        quality.to_csv(RESULTS/'coffee_shop_quality.csv',index=False)
    return valid,orders,items,products,quality,rejected


def sales_summary(valid, start=None,end=None,store=None):
    d=valid
    if start: d=d[d.date>=str(start)]
    if end: d=d[d.date<=str(end)]
    if store and store!='All stores': d=d[d.store_location.str.casefold()==str(store).casefold()]
    gross=round(float(d.gross_line_amount.sum()),2)
    n=int(d.order_id.nunique())
    return {'start':start,'end':end,'store':store or 'All stores','currency':'USD (assumed from NYC location; not explicit in source)',
            'gross_transaction_revenue':gross,'transaction_count':n,'units_sold':int(d.quantity.sum()),
            'gross_aov':round(gross/n,2) if n else None,
            'note':'Gross transaction revenue only; discounts, refunds and verified net sales are unavailable.'}


def top_products_real(valid,start=None,end=None,store=None,metric='units'):
    d=valid
    if start:d=d[d.date>=str(start)]
    if end:d=d[d.date<=str(end)]
    if store and store!='All stores':d=d[d.store_location.str.casefold()==str(store).casefold()]
    ranked=d.groupby(['product_id','product_detail'],as_index=False).agg(units=('quantity','sum'),gross_revenue=('gross_line_amount','sum'))
    key='gross_revenue' if metric=='revenue' else 'units'
    return ranked.sort_values([key,'product_id'],ascending=[False,True]).head(5)


def build_real_reports(valid,quality):
    """Before/after charts, engineered features, and a time-split RF experiment."""
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import r2_score,mean_absolute_error
    for p in (FIGURES,RESULTS,OUTPUT):p.mkdir(parents=True,exist_ok=True)
    daily=(valid.groupby('date',as_index=False).agg(gross_transaction_revenue=('gross_line_amount','sum'),
        transaction_count=('order_id','nunique'),units_sold=('quantity','sum'),
        avg_unit_price=('unit_price','mean'),avg_hour=('hour_of_day','mean')))
    daily['day_of_week']=pd.to_datetime(daily.date).dt.dayofweek
    daily['month']=pd.to_datetime(daily.date).dt.month
    daily.to_csv(OUTPUT/'daily_sales_features.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4))
    ax.bar(['Raw rows','Validated rows'],[int(quality.raw_rows.iloc[0]),int(quality.valid_rows.iloc[0])]);ax.set_ylabel('Rows');ax.set_title('Coffee Shop Sales: raw vs validated')
    fig.tight_layout();fig.savefig(FIGURES/'coffee_shop_before_after.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4));ax.plot(pd.to_datetime(daily.date),daily.gross_transaction_revenue)
    ax.set_title('Real coffee shop daily gross transaction revenue');ax.set_ylabel('Revenue (USD assumed)');fig.autofmt_xdate();fig.tight_layout();fig.savefig(FIGURES/'coffee_shop_daily_revenue.png',dpi=150);plt.close(fig)
    # Categorical encoding + scaling demonstrated without leaking future test data.
    from sklearn.preprocessing import StandardScaler
    categorical=pd.get_dummies(valid[['store_location','product_category']],dtype=int)
    numeric=valid[['quantity','unit_price','hour_of_day']].astype(float)
    transformed=pd.concat([numeric.reset_index(drop=True),categorical.reset_index(drop=True)],axis=1)
    # Descriptive encoded artifact only; model training below fits its own train split.
    transformed.to_csv(OUTPUT/'encoded_features_unscaled.csv',index=False)
    columns=['transaction_count','units_sold','avg_unit_price','avg_hour','day_of_week','month']
    cut=int(len(daily)*.75)
    if cut<10 or len(daily)-cut<5:raise ValueError('Insufficient days for chronological split')
    train,test=daily.iloc[:cut],daily.iloc[cut:]
    scaler=StandardScaler().fit(train[columns])
    Xtrain=scaler.transform(train[columns]);Xtest=scaler.transform(test[columns])
    model=RandomForestRegressor(n_estimators=100,min_samples_leaf=2,random_state=42,n_jobs=-1)
    model.fit(Xtrain,train.gross_transaction_revenue)
    pred=model.predict(Xtest)
    pi=permutation_importance(model,Xtest,test.gross_transaction_revenue,n_repeats=5,random_state=42,scoring='neg_mean_absolute_error',n_jobs=1)
    imp=pd.DataFrame({'feature':columns,'importance_mae_increase':pi.importances_mean,'std':pi.importances_std}).sort_values('importance_mae_increase',ascending=False)
    imp.to_csv(RESULTS/'coffee_shop_feature_importance.csv',index=False)
    metrics={'train_days':len(train),'test_days':len(test),'test_r2':float(r2_score(test.gross_transaction_revenue,pred)),
             'test_mae':float(mean_absolute_error(test.gross_transaction_revenue,pred)),
             'baseline_mae':float(mean_absolute_error(test.gross_transaction_revenue,np.repeat(train.gross_transaction_revenue.mean(),len(test))))}
    pd.DataFrame([metrics]).to_csv(RESULTS/'coffee_shop_model_metrics.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4));imp.sort_values('importance_mae_increase').plot.barh(x='feature',y='importance_mae_increase',ax=ax,legend=False)
    ax.set_xlabel('Permutation importance: increase in MAE');ax.set_title('Real coffee shop: feature importance');fig.tight_layout();fig.savefig(FIGURES/'coffee_shop_feature_importance.png',dpi=150);plt.close(fig)
    return daily,imp,metrics


class RealCoffeeAssistant:
    """Real transaction analytics + existing FAQ retriever; no invented net sales."""
    def __init__(self,valid,retriever):self.valid=valid;self.retriever=retriever
    def answer(self,question,store='All stores'):
        import re
        q=question.strip(); low=q.lower()
        if not q:return 'Please enter a question.'
        if any(x in low for x in ['net sales','refund','discount','profit','forecast','predict','payment method']):
            if 'what is' not in low and 'define' not in low and 'difference' not in low:
                return 'Unavailable in the real Coffee Shop Sales file: discounts, refunds, verified net sales, profit, forecasts, and payment methods. Try gross sales, top products, or hourly sales.'
        dates=re.findall(r'20\d{2}-\d{2}-\d{2}',q)
        start,end=(dates[0],dates[-1]) if dates else (None,None)
        # Natural-language month support for source's 2023 Jan-June period
        months={'january':1,'february':2,'march':3,'april':4,'may':5,'june':6}
        for word,m in months.items():
            if word in low and not dates:
                start=f'2023-{m:02d}-01';end=str((pd.Timestamp(start)+pd.offsets.MonthEnd(0)).date())
                break
        if 'top' in low and any(t in low for t in ['product','item','selling']):
            metric='revenue' if 'revenue' in low else 'units'
            return f'Top 5 products by {metric} ({start or "all dates"} to {end or "all dates"}; {store}):\n'+top_products_real(self.valid,start,end,store,metric).to_string(index=False)
        if any(t in low for t in ['peak hour','busiest hour','hourly']):
            d=self.valid
            if start:d=d[d.date>=start]
            if end:d=d[d.date<=end]
            if store!='All stores':d=d[d.store_location==store]
            counts=d.groupby('hour_of_day').order_id.nunique().sort_values(ascending=False)
            return 'No transactions for selected period.' if counts.empty else f'Busiest hour: {int(counts.index[0]):02d}:00 ({int(counts.iloc[0]):,} transactions; {store}).'
        if not (low.startswith('what is ') or low.startswith('define ') or low.startswith('explain ') or 'difference between' in low) and any(t in low for t in ['sales','revenue','transaction','aov','average order value','units sold']):
            s=sales_summary(self.valid,start,end,store)
            return (f"Store: {s['store']} | Period: {start or 'all dates'} to {end or 'all dates'}\n"
                    f"Gross transaction revenue: ${s['gross_transaction_revenue']:,.2f} (USD assumed)\n"
                    f"Transactions: {s['transaction_count']:,} | Units: {s['units_sold']:,}\n"
                    f"Gross AOV: {'N/A' if s['gross_aov'] is None else '$'+format(s['gross_aov'],',.2f')}\n{s['note']}")
        hits=self.retriever.search(q)
        if not hits or hits[0]['score']<.20:return 'No relevant business definition found. Try asking about gross sales or product rankings.'
        h=hits[0]
        return f"{h['content']}\n\nKnowledge source: {h['doc_id']} ({h['source']}); this definition is not a calculated Kaggle metric."
