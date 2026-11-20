"""Controlled corruption of copied raw data; production raw CSVs remain untouched."""
from pathlib import Path
import pandas as pd
from .generate import RAW

def inject_dirty_data(target):
    target=Path(target);target.mkdir(parents=True,exist_ok=True)
    for name in ['orders','order_items','products','knowledge_base']:
        pd.read_csv(RAW/f'{name}.csv').to_csv(target/f'{name}.csv',index=False)
    o=pd.read_csv(target/'orders.csv');i=pd.read_csv(target/'order_items.csv');p=pd.read_csv(target/'products.csv')
    o.loc[0,'order_date']=None
    o.loc[1,'discount_total']=-5
    o=pd.concat([o,o.iloc[[2]]],ignore_index=True)
    i.loc[0,'quantity']=-2
    i.loc[1,'product_id']=999999
    i=pd.concat([i,i.iloc[[2]]],ignore_index=True)
    p.loc[0,'category']=None
    o.to_csv(target/'orders.csv',index=False);i.to_csv(target/'order_items.csv',index=False);p.to_csv(target/'products.csv',index=False)
    return target
