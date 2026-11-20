"""Reproducible 50-question evaluation with explicit gold answers."""
import time
import numpy as np
import pandas as pd
from .analytics import daily_metrics
from .evaluate import baseline_tfidf
from .generate import BASE

def build_questions(completed):
    # Gold references are generated from the metric dictionary and independent reference aggregations.
    from .knowledge import TOPICS
    knowledge=[]
    for j,(title,content) in enumerate(TOPICS[:20],1):
        knowledge.append({'kind':'knowledge','question':f'Explain {title.lower()} in this POS system','gold_doc_id':f'KB{j:03d}','expected':content})
    dates=sorted(completed.date.unique())[:20]
    numeric=[]
    for d in dates:
        m=daily_metrics(completed,d)
        numeric.append({'kind':'numeric','question':f'What were net sales on {d}?','date':d,'expected_net':m['net_sales']})
    unsupported=[{'kind':'unsupported','question':q} for q in [
        'Can you predict next month profit?', 'Can you run payroll?', 'Can you file my taxes?',
        'Can you process my credit card number?', 'Can you forecast next quarter revenue?',
        'Can you predict tomorrow sales?', 'Can you handle tax filing?',
        'Can you access customer card numbers?', 'Can you calculate next month profit?',
        'Can you predict future revenue?']]
    return pd.DataFrame(knowledge+numeric+unsupported)

def run_evaluation(bot,kb,completed,output_dir):
    output_dir.mkdir(parents=True,exist_ok=True)
    q=build_questions(completed);q.to_csv(output_dir/'evaluation_questions_50.csv',index=False)
    records=[]
    for _,row in q.iterrows():
        start=time.perf_counter();answer=bot.answer(row.question);elapsed=time.perf_counter()-start
        kind=row.kind
        if kind=='knowledge':
            hits=bot.retriever.search(row.question,3)
            correct=any(h['doc_id']==row.gold_doc_id for h in hits)
            supported=f'Source: {row.gold_doc_id}' in answer and row.expected in answer
            records.append({'kind':kind,'question':row.question,'correct':correct,'grounded':supported,'cited':supported,'latency_seconds':elapsed})
        elif kind=='numeric':
            actual=daily_metrics(completed,row.date)['net_sales']
            # independently recompute from completed order rows, not from the assistant response
            subset=completed[completed.date==row.date]
            independent=round(float((subset.gross_sales-subset.discount_total-subset.refund_total).sum()),2)
            correct=abs(actual-independent)<0.011 and f'Net sales: ${actual:.2f}' in answer
            records.append({'kind':kind,'question':row.question,'correct':correct,'grounded':correct,'cited':pd.NA,'latency_seconds':elapsed})
        else:
            correct=answer.startswith('Unsupported request:')
            records.append({'kind':kind,'question':row.question,'correct':correct,'grounded':pd.NA,'cited':pd.NA,'latency_seconds':elapsed})
    r=pd.DataFrame(records);r.to_csv(output_dir/'evaluation_results_50.csv',index=False)
    knowledge=r[r.kind=='knowledge'];numeric=r[r.kind=='numeric'];unsupported=r[r.kind=='unsupported']
    pairs=list(zip(q[q.kind=='knowledge'].question,q[q.kind=='knowledge'].gold_doc_id))
    tfidf=baseline_tfidf(kb,pairs)
    metrics={
      'retrieval_recall_at_3':float(knowledge.correct.mean()),
      'tfidf_baseline_recall_at_3':float(tfidf),
      'numeric_correctness':float(numeric.correct.mean()),
      'grounded_answer_accuracy':float(knowledge.grounded.mean()),
      'citation_support':float(knowledge.cited.mean()),
      'unsupported_query_handling':float(unsupported.correct.mean()),
      'median_response_latency_seconds':float(r.latency_seconds.median()),
      'total_questions':len(r)}
    pd.DataFrame([metrics]).to_csv(output_dir/'evaluation_summary.csv',index=False)
    return metrics,r
