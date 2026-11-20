import pandas as pd
from bizassist.coffee_shop import load_and_prepare,sales_summary,top_products_real,RealCoffeeAssistant


def test_real_source_and_revenue():
    valid,orders,items,products,quality,rejected=load_and_prepare(save=False)
    assert len(valid)==149116
    assert len(rejected)==0
    assert len(products)==80
    assert len(orders)==len(items)==len(valid)
    assert set(valid.store_location)=={'Lower Manhattan',"Hell's Kitchen",'Astoria'}
    summary=sales_summary(valid)
    independent=(valid.transaction_qty*valid.unit_price).sum()
    assert abs(summary['gross_transaction_revenue']-round(independent,2))<.001
    assert summary['transaction_count']==valid.transaction_id.nunique()
    assert 'net_sales' not in summary


def test_store_filter_and_rankings():
    valid,*_=load_and_prepare(save=False)
    filtered=valid[(valid.store_location=='Astoria')&(valid.date>='2023-03-01')&(valid.date<='2023-03-31')]
    actual=sales_summary(valid,'2023-03-01','2023-03-31','Astoria')
    assert actual['transaction_count']==len(filtered)
    assert abs(actual['gross_transaction_revenue']-round((filtered.transaction_qty*filtered.unit_price).sum(),2))<.001
    top=top_products_real(valid,'2023-03-01','2023-03-31','Astoria')
    assert len(top)==5
    assert top.units.is_monotonic_decreasing


def test_rejects_invalid_rows(tmp_path):
    sample=pd.DataFrame([
        [1,'2023-01-01','08:00:00',2,5,'Lower Manhattan',32,3.0,'Coffee','Brewed','Ethiopia'],
        [2,'2023-01-01','09:00:00',-1,5,'Lower Manhattan',32,3.0,'Coffee','Brewed','Ethiopia']
    ],columns=['transaction_id','transaction_date','transaction_time','transaction_qty','store_id','store_location','product_id','unit_price','product_category','product_type','product_detail'])
    path=tmp_path/'mini.xlsx';sample.to_excel(path,index=False)
    valid,*_,rejected=load_and_prepare(path,save=False)
    assert len(valid)==1 and len(rejected)==1
    assert 'invalid_quantity' in rejected.rejection_reason.iloc[0]
