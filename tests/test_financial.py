import pandas as pd
from bizassist.analytics import prepare,daily_metrics,top_products

def sample():
    o=pd.DataFrame({'order_id':[1,2,3], 'order_date':['2026-11-08']*3,
                    'status':['completed','completed','cancelled'],'discount_total':[2.,0.,0.], 'refund_total':[3.,10.,0.]})
    i=pd.DataFrame({'item_id':[1,2,3],'order_id':[1,2,3],'product_id':[1,1,1], 'quantity':[2,1,1],'unit_price':[10.,10.,10.]})
    p=pd.DataFrame({'product_id':[1],'product_name':['Tea'],'category':['Drink']})
    return prepare(o,i,p)

def test_net_and_aov():
    c,e=sample(); m=daily_metrics(c,'2026-11-08')
    assert m['gross_sales']==30
    assert m['net_sales']==15
    assert m['eligible_order_count']==2
    assert m['aov']==7.5

def test_zero_orders():
    c,_=sample();m=daily_metrics(c,'2026-11-09');assert m['aov'] is None and m['net_sales']==0

def test_product_ranking_excludes_cancelled():
    _,e=sample(); assert top_products(e,'2026-11-08','2026-11-08').iloc[0]['units']==3
