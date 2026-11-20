import pandas as pd

def prepare(orders,items,products):
    o=orders.copy(); i=items.copy(); o['order_date']=pd.to_datetime(o.order_date)
    o['date']=o.order_date.dt.date.astype(str)
    o['day_of_week']=o.order_date.dt.dayofweek
    o['hour_of_day']=o.order_date.dt.hour
    i['gross_line_amount']=i.quantity*i.unit_price
    completed=o[o.status=='completed'].copy()
    eligible=i.merge(completed[['order_id','date','day_of_week','hour_of_day']],on='order_id',how='inner')
    eligible=eligible.merge(products[['product_id','product_name','category']],on='product_id',how='left')
    order_gross=eligible.groupby('order_id').gross_line_amount.sum()
    completed['gross_sales']=completed.order_id.map(order_gross).fillna(0)
    completed['net_sales']=completed.gross_sales-completed.discount_total-completed.refund_total
    return completed,eligible

def daily_metrics(completed,date):
    rows=completed[completed.date==str(date)]
    count=len(rows); gross=round(float(rows.gross_sales.sum()),2)
    discount=round(float(rows.discount_total.sum()),2); refunds=round(float(rows.refund_total.sum()),2)
    net=round(gross-discount-refunds,2)
    return {'date':str(date),'gross_sales':gross,'discounts':discount,'refunds':refunds,'net_sales':net,
            'eligible_order_count':count,'aov':round(net/count,2) if count else None}

def top_products(eligible,start=None,end=None,metric='units'):
    rows=eligible.copy()
    if start: rows=rows[rows.date>=str(start)]
    if end: rows=rows[rows.date<=str(end)]
    g=rows.groupby(['product_id','product_name'],as_index=False).agg(units=('quantity','sum'),gross_revenue=('gross_line_amount','sum'))
    col='units' if metric=='units' else 'gross_revenue'
    return g.sort_values([col,'product_id'],ascending=[False,True]).head(5)
