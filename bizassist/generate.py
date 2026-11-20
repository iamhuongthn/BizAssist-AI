from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / 'data' / 'raw'

def generate(seed=42, n_orders=2000):
    RAW.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    categories = ['Milk Tea', 'Coffee', 'Tea', 'Snack', 'Topping']
    names = [f'{categories[i%5]} Item {i+1:02d}' for i in range(30)]
    products = pd.DataFrame({'product_id': range(1,31), 'product_name':names,
                             'category':[categories[i%5] for i in range(30)]})
    prices = np.round(rng.uniform(2.5, 9.5, 30),2)
    start = pd.Timestamp('2026-10-01')
    dates = start + pd.to_timedelta(rng.integers(0,61,n_orders),unit='D') + pd.to_timedelta(rng.integers(8,22,n_orders),unit='h')
    statuses = rng.choice(['completed','cancelled','failed'],n_orders,p=[.90,.07,.03])
    orders=[]; items=[]; iid=1
    for oid in range(1,n_orders+1):
        status=str(statuses[oid-1]); count=int(rng.integers(1,5))
        chosen=rng.choice(np.arange(1,31),size=count,replace=True,p=np.array([31-i for i in range(1,31)])/sum(range(1,31)))
        gross=0
        for pid in chosen:
            qty=int(rng.integers(1,4)); price=float(prices[int(pid)-1]); gross+=qty*price
            items.append((iid,oid,int(pid),qty,price)); iid+=1
        discount=round(gross*float(rng.choice([0,.05,.1],p=[.7,.2,.1])),2) if status=='completed' else 0.0
        refund=round((gross-discount)*float(rng.choice([0,.25,1],p=[.92,.06,.02])),2) if status=='completed' else 0.0
        orders.append((oid,dates[oid-1],status,discount,refund))
    pd.DataFrame(orders,columns=['order_id','order_date','status','discount_total','refund_total']).to_csv(RAW/'orders.csv',index=False)
    pd.DataFrame(items,columns=['item_id','order_id','product_id','quantity','unit_price']).to_csv(RAW/'order_items.csv',index=False)
    products.to_csv(RAW/'products.csv',index=False)
    from .knowledge import build_knowledge
    build_knowledge(RAW / 'knowledge_base.csv')
    return {'orders':len(orders),'items':len(items),'products':len(products),'knowledge':40}

if __name__=='__main__': print(generate())
