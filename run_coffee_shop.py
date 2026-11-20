"""Standalone real-data workflow; leaves original synthetic project untouched."""
from pathlib import Path
from bizassist.coffee_shop import load_and_prepare,build_real_reports,RealCoffeeAssistant,sales_summary
from bizassist.preprocess import process
from bizassist.retrieval import KnowledgeRetriever
from bizassist.generate import BASE
import pandas as pd

def main(use_embeddings=True):
    valid,orders,items,products,quality,rejected=load_and_prepare()
    daily,importance,metrics=build_real_reports(valid,quality)
    # FAQ is sourced from the original project's existing curated knowledge base.
    kb_path=BASE/'data'/'raw'/'knowledge_base.csv'
    if not kb_path.exists():
        from bizassist.knowledge import build_knowledge
        kb_path.parent.mkdir(parents=True,exist_ok=True);build_knowledge(kb_path)
    kb=pd.read_csv(kb_path)
    bot=RealCoffeeAssistant(valid,KnowledgeRetriever(kb,use_embeddings=use_embeddings))
    examples=['What were gross sales in March 2023?',
              'Show top five products by units sold from 2023-03-01 to 2023-03-31',
              'What was the busiest hour?', 'What is gross sales?',
              'What were net sales in March 2023?']
    with open(BASE/'reports'/'results'/'coffee_shop_example_interactions.txt','w') as f:
        for q in examples:
            a=bot.answer(q);f.write('Q: '+q+'\nA: '+a+'\n\n')
    print('REAL DATA QUALITY\n',quality.to_string(index=False))
    print('REAL MODEL EVALUATION',metrics)
    print('REAL DATA SALES SUMMARY',sales_summary(valid))
    return bot

if __name__=='__main__':main(use_embeddings=False)
