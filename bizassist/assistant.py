import re
from datetime import date,timedelta
from .analytics import daily_metrics,top_products

class BizAssist:
    def __init__(self,completed,eligible,retriever,reference_date=None):
        self.completed=completed; self.eligible=eligible; self.retriever=retriever
        self.reference_date=date.fromisoformat(str(reference_date)) if reference_date else date.today()

    def answer(self,question):
        q=question.strip(); low=q.lower()
        if not q: return 'Please enter a question.'
        if any(s in low for s in ['predict','forecast','payroll','tax filing','credit card number','next month profit','file my taxes','card numbers']):
            return 'Unsupported request: this prototype provides historical synthetic sales analytics and documented POS definitions, not forecasting, payroll, tax or payment processing.'
        match=re.search(r'\b20\d{2}-\d{2}-\d{2}\b',q)
        day=match.group(0) if match else None
        if not day and 'yesterday' in low: day=str(self.reference_date-timedelta(days=1))
        if not day and 'today' in low: day=str(self.reference_date)
        if 'top' in low and any(w in low for w in ['product','item','selling']):
            if 'last week' in low: return 'Please specify an explicit start and end date (YYYY-MM-DD) for reproducible product rankings.'
            dates=re.findall(r'\b20\d{2}-\d{2}-\d{2}\b',q)
            if not dates: return 'Please provide a reporting date or date range in YYYY-MM-DD format.'
            metric='revenue' if 'revenue' in low else 'units'
            table=top_products(self.eligible,dates[0],dates[-1],metric)
            return f'Top products by {metric} for {dates[0]} to {dates[-1]} (completed orders):\n\n'+table.to_string(index=False)
        if any(s in low for s in ['net sales','gross sales','average order value','aov','eligible order','refund total','refunds on','refund on','discount total']) and (day or 'sales on' in low):
            if not day: return 'Please provide a date (YYYY-MM-DD) for the sales calculation.'
            d=daily_metrics(self.completed,day)
            return (f"Date: {day}\nGross sales: ${d['gross_sales']:.2f}\nDiscounts: ${d['discounts']:.2f}\n"
                    f"Refunds: ${d['refunds']:.2f}\nNet sales: ${d['net_sales']:.2f}\n"
                    f"Eligible completed orders: {d['eligible_order_count']}\n"
                    f"AOV: {'N/A' if d['aov'] is None else '$'+format(d['aov'],'.2f')}\n"
                    'Formula: net sales = gross sales - discounts - refunds; AOV = net sales / eligible orders.')
        hits=self.retriever.search(q)
        if not hits or hits[0]['score']<0.20:
            return 'I could not find sufficiently relevant evidence in the knowledge base. Please clarify or ask about POS sales metrics.'
        h=hits[0]
        return f"{h['content']}\n\nSource: {h['doc_id']} ({h['source']}; updated {h['updated_at']})\nRetrieval: {self.retriever.mode}, score {h['score']:.3f}"
