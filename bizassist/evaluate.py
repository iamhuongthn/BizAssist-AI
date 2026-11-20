import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from .generate import BASE
from .analytics import daily_metrics

def evaluate_retrieval(retriever,questions):
    scores=[]; times=[]
    for q,gold in questions:
        t=time.perf_counter(); hits=retriever.search(q,3); times.append(time.perf_counter()-t)
        scores.append(any(h['doc_id']==gold for h in hits))
    return {'recall_at_3':float(np.mean(scores)),'median_retrieval_seconds':float(np.median(times)),'n_questions':len(scores)}

def baseline_tfidf(knowledge,questions):
    texts=(knowledge.title.fillna('')+' '+knowledge.content.fillna('')).tolist()
    v=TfidfVectorizer(ngram_range=(1,2),stop_words='english'); m=v.fit_transform(texts)
    found=[]
    for q,gold in questions:
        scores=cosine_similarity(v.transform([q]),m).ravel()
        found.append(gold in knowledge.iloc[np.argsort(-scores)[:3]].doc_id.values)
    return float(np.mean(found))

def feature_importance(completed,output=None):
    # Exploratory prediction of daily net sales. Do not include gross sales or net sales as input features.
    daily=completed.groupby('date',as_index=False).agg(net_sales=('net_sales','sum'),order_count=('order_id','count'),
        discount_total=('discount_total','sum'),refund_total=('refund_total','sum'),avg_hour=('hour_of_day','mean'))
    daily['day_of_week']=pd.to_datetime(daily.date).dt.dayofweek
    cols=['order_count','discount_total','refund_total','avg_hour','day_of_week']
    if len(daily)<15: raise ValueError('Need at least 15 daily observations')
    xtrain,xtest,ytrain,ytest=train_test_split(daily[cols],daily.net_sales,test_size=.25,random_state=42,shuffle=False)
    model=RandomForestRegressor(n_estimators=100,random_state=42,min_samples_leaf=2).fit(xtrain,ytrain)
    result=permutation_importance(model,xtest,ytest,n_repeats=10,random_state=42,scoring='neg_mean_absolute_error')
    frame=pd.DataFrame({'feature':cols,'importance':result.importances_mean,'std':result.importances_std}).sort_values('importance',ascending=False)
    if output: frame.to_csv(output,index=False)
    return frame,model.score(xtest,ytest)
