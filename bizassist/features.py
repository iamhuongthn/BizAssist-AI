"""Explicit encoded and standardized features for coursework evidence."""
import pandas as pd
from sklearn.preprocessing import StandardScaler

def transform_features(orders,items,products,output_dir):
    output_dir.mkdir(parents=True,exist_ok=True)
    o=orders.copy();o['order_date']=pd.to_datetime(o.order_date)
    o['day_of_week']=o.order_date.dt.dayofweek;o['hour_of_day']=o.order_date.dt.hour
    numeric=['discount_total','refund_total','day_of_week','hour_of_day']
    scaler=StandardScaler()
    scaled=pd.DataFrame(scaler.fit_transform(o[numeric]),columns=[f'{c}_scaled' for c in numeric],index=o.index)
    encoded=pd.get_dummies(o[['status']],prefix=['status'],dtype=int)
    out=pd.concat([o[['order_id']],scaled,encoded],axis=1)
    out.to_csv(output_dir/'encoded_scaled_order_features.csv',index=False)
    return out
