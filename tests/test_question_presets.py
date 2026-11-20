from run_unified import QUESTION_PRESETS, preset_choices, UnifiedBizAssist

class Stub:
    def __init__(self, real=False):
        self.real=real
        if real:
            import pandas as pd
            self.valid=pd.DataFrame({'store_location':["Hell's Kitchen",'Astoria']})
    def answer(self, question, store=None):
        return 'real answer' if self.real else 'synthetic answer'

def app():return UnifiedBizAssist(Stub(),Stub(real=True))

def test_presets_unique_and_contextual():
    questions=[q for _,q in QUESTION_PRESETS]
    assert questions[0]=='' and len(questions)>10
    assert len(set(questions))==len(questions)
    assert preset_choices()==QUESTION_PRESETS

def test_automatic_routing():
    a=app()
    assert a.route_question('Gross sales in March 2023?')[0]=='real'
    assert a.route_question('Refunds on 2026-11-03?')[0]=='synthetic'
    assert a.route_question('What is net sales?')[0]=='knowledge'
    assert a.route_question('What were total sales?')[0]=='clarify'
    assert a.route_question('What were net sales in March 2023?')[0]=='real'
    assert a.route_question('Sales 2026-11-08?',store="Hell's Kitchen")[0]=='clarify'
    assert 'Real Coffee Shop' in a.answer('Gross sales in March 2023?')
    assert 'Synthetic POS' in a.answer('Refunds on 2026-11-03?')
    assert 'knowledge base' in a.answer('What is net sales?')

def test_legacy_manual_evaluation_api():
    a=app()
    assert a.answer('x',dataset='Synthetic POS')=='synthetic answer'
    assert a.answer('x',dataset='Real Coffee Shop (Kaggle)')=='real answer'
