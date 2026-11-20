from pathlib import Path
import sys
import pandas as pd
from bizassist.generate import generate,BASE
from bizassist.preprocess import process
from bizassist.analytics import prepare,daily_metrics
from bizassist.retrieval import KnowledgeRetriever
from bizassist.assistant import BizAssist
from bizassist.evaluate import feature_importance,baseline_tfidf,evaluate_retrieval
from bizassist.audit import inject_dirty_data
from bizassist.features import transform_features
from bizassist.evaluation_full import run_evaluation

def main(use_embeddings=True,launch=False):
    generate(); orders,items,products,kb,quality=process()
    transform_features(orders,items,products,BASE/'data'/'processed')
    completed,eligible=prepare(orders,items,products)
    completed.to_csv(BASE/'data'/'processed'/'completed_orders_features.csv',index=False)
    eligible.to_csv(BASE/'data'/'processed'/'eligible_item_features.csv',index=False)
    print('DATA QUALITY\n',quality.to_string(index=False))
    print('EXAMPLE DAILY METRICS',daily_metrics(completed,'2026-11-08'))
    figures=BASE/'reports'/'figures'; figures.mkdir(parents=True,exist_ok=True)
    import matplotlib.pyplot as plt
    quality.set_index('table')[['before_rows','after_rows']].plot(kind='bar',title='Raw vs Validated Row Counts')
    plt.tight_layout();plt.savefig(figures/'before_after_rows.png');plt.close()
    importance,r2=feature_importance(completed,BASE/'reports'/'results'/'feature_importance.csv')
    importance.set_index('feature').importance.plot(kind='barh',title='Permutation Importance (held-out MAE increase)')
    plt.tight_layout();plt.savefig(figures/'feature_importance.png');plt.close()
    print('FEATURE IMPORTANCE\n',importance.to_string(index=False),'\nHoldout R2:',round(r2,3))
    retriever=KnowledgeRetriever(kb,use_embeddings=use_embeddings)
    bot=BizAssist(completed,eligible,retriever,reference_date="2026-11-09")
    questions=[('What is gross sales?', 'KB001'),('Explain net sales','KB002'),('What is AOV?', 'KB003'),
               ('Which orders count as eligible?', 'KB004'),('Are cancelled orders included?', 'KB005')]
    metrics=evaluate_retrieval(retriever,questions)
    metrics['tfidf_baseline_recall_at_3']=baseline_tfidf(kb,questions)
    pd.DataFrame([metrics]).to_csv(BASE/'reports'/'results'/'retrieval_smoke_test.csv',index=False)
    print('RETRIEVAL SMOKE TEST (NOT FINAL EVALUATION)',metrics)
    examples=['What is net sales?','What were my net sales on 2026-11-08?',
              'What was the average order value on 2026-11-08?',
              'Show top five products by units sold from 2026-11-01 to 2026-11-08',
              'Can you predict next month profit?']
    with open(BASE/'reports'/'results'/'example_interactions.txt','w') as f:
        for q in examples:
            response=bot.answer(q);f.write('Q: '+q+'\nA: '+response+'\n\n');print('\nQ:',q,'\nA:',response)
    full_metrics, full_results=run_evaluation(bot,kb,completed,BASE/'reports'/'results')
    print('FULL 50-QUESTION EVALUATION',full_metrics)
    # Controlled dirty-data audit after the final clean model has been prepared.
    audit_path=inject_dirty_data(BASE/'data'/'audit_raw')
    _,_,_,_,dirty_quality=process(raw_dir=audit_path)
    dirty_quality.to_csv(BASE/'reports'/'results'/'dirty_data_quality.csv',index=False)
    print('CONTROLLED DIRTY-DATA AUDIT\n',dirty_quality.to_string(index=False))
    # Restore the clean processed artifacts for reproducible subsequent runs.
    process()
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    dirty_quality.set_index('table')[['before_rows','after_rows']].plot(kind='bar',ax=axes[0],title='Controlled dirty data: before vs after')
    dirty_quality.set_index('table')[['before_missing_cells','after_missing_cells']].plot(kind='bar',ax=axes[1],title='Missing cells: before vs after')
    plt.tight_layout();plt.savefig(figures/'dirty_data_before_after.png');plt.close()
    if launch:
        import gradio as gr
        ui=gr.Interface(fn=bot.answer,inputs=gr.Textbox(label='Ask a POS business question',lines=2),
            outputs=gr.Textbox(label='BizAssist AI response',lines=12),title='BizAssist AI',examples=examples)
        ui.launch(share=False)
    return bot

if __name__=='__main__': main(use_embeddings='--tfidf' not in sys.argv,launch='--ui' in sys.argv)
