"""Unified entry point for synthetic and real coffee-shop datasets."""
from run import main as synthetic_main
from run_coffee_shop import main as real_main
from bizassist.coffee_shop import sales_summary

import re
from datetime import date

class UnifiedBizAssist:
    """Routes queries without combining real and simulated financial records.

    dataset= is retained for backward compatibility with the 30-case evaluator.
    User-facing Gradio calls omit dataset and rely on route_question().
    """
    def __init__(self, synthetic_bot, real_bot):
        self.synthetic_bot = synthetic_bot
        self.real_bot = real_bot

    @property
    def stores(self):
        return ["All stores"] + sorted(self.real_bot.valid.store_location.dropna().unique().tolist())

    def route_question(self, question, store="All stores"):
        q = (question or "").strip()
        if not q:
            return "clarify", "Please enter a business question."
        low = q.lower()
        # Definitions do not depend on transaction datasets.
        if re.match(r"^(what is|what are|define|explain|difference between|what's)\b", low) and not re.search(r"\b20\d{2}\b", low) and not any(t in low for t in ("yesterday", "today", "last week", "this month")):
            return "knowledge", ""
        years = set(re.findall(r"\b(20\d{2})\b", low))
        mentions_real_store = any(name.lower() in low for name in self.stores if name != "All stores")
        store_selected = store != "All stores"
        # A store filter means a real-data request; never fabricate refund or net figures.
        if len(years) > 1 or ("2023" in years and "2026" in years):
            return "clarify", "Your question references multiple data periods. Please ask about one reporting period at a time."
        if "2026" in years and (mentions_real_store or store_selected):
            return "clarify", "The selected coffee-shop store is available only for 2023 real data. Clear the store selection for synthetic 2026 data."
        if "2023" in years or mentions_real_store or store_selected:
            return "real", ""
        if "2026" in years or "yesterday" in low or "today" in low:
            return "synthetic", ""
        if any(t in low for t in ("march", "april", "january", "february", "june", "may", "hell's kitchen", "astoria", "lower manhattan", "busiest hour", "peak hour", "hourly")):
            return "real", ""
        if any(t in low for t in ("refund", "discount", "net sales", "eligible order")):
            return "clarify", "For refunds, discounts or net sales, specify a synthetic reporting date (YYYY-MM-DD in 2026). Real Kaggle data does not contain these fields."
        if any(t in low for t in ("sales", "revenue", "top product", "top five", "top 5", "average order value", "aov", "transaction", "units sold")):
            return "clarify", "Please specify a reporting period: 2023 for real Coffee Shop data or a 2026 date for synthetic POS data. The sources are never combined."
        return "knowledge", ""

    def answer(self, question, dataset=None, store="All stores"):
        # Legacy explicit-dataset API supports existing evaluation without changing its gold labels.
        if dataset in ("Synthetic POS", "Real Coffee Shop (Kaggle)"):
            result = (self.synthetic_bot.answer(question) if dataset == "Synthetic POS"
                      else self.real_bot.answer(question, store=store))
            return result
        route, message = self.route_question(question, store)
        if route == "clarify":
            return message
        if route == "real":
            result = self.real_bot.answer(question, store=store)
            return result + "\n\nData source: Real Coffee Shop Sales (Kaggle, 2023)."
        if route == "synthetic":
            result = self.synthetic_bot.answer(question)
            return result + "\n\nData source: Synthetic POS transactions (2026; simulated)."
        result = self.synthetic_bot.answer(question)
        return result + "\n\nData source: Project-authored POS knowledge base (definitions; not sales records)."

def main(use_embeddings=True):
    print("=== 1/2: SYNTHETIC POS PIPELINE ===")
    synthetic_bot = synthetic_main(use_embeddings=use_embeddings, launch=False)
    print("=== 2/2: REAL COFFEE SHOP PIPELINE ===")
    real_bot = real_main(use_embeddings=use_embeddings)
    return UnifiedBizAssist(synthetic_bot, real_bot)


# One common-question list. Each question provides enough context to route safely.
QUESTION_PRESETS = [
    ("Choose a common question…", ""),
    ("Real · Gross sales : March 2023", "What were gross sales in March 2023?"),
    ("Real · Gross sales : April 2023", "What was gross revenue in April 2023?"),
    ("Real · Top 5 products by units : March", "Show top five products by units sold from 2023-03-01 to 2023-03-31"),
    ("Real · Top 5 products by revenue : March", "Show top five products by revenue from 2023-03-01 to 2023-03-31"),
    ("Real · Busiest transaction hour", "What was the busiest hour in March 2023?"),
    ("Real · Gross AOV : March", "What was the average order value in March 2023?"),
    ("Real · Unavailable net sales example", "What were net sales in March 2023?"),
    ("Synthetic · Net sales : Nov 8", "What were net sales on 2026-11-08?"),
    ("Synthetic · Gross sales : Nov 8", "What were gross sales on 2026-11-08?"),
    ("Synthetic · Refunds : Nov 3", "What was the refund total on 2026-11-03?"),
    ("Synthetic · Discounts : Nov 8", "What was the discount total on 2026-11-08?"),
    ("Synthetic · Top 5 products", "Show top five products by units sold from 2026-11-01 to 2026-11-08"),
    ("Knowledge · Definition of net sales", "What is net sales?"),
    ("Knowledge · Definition of AOV", "What is average order value?"),
    ("Clarification example · No reporting period", "What were total sales?"),
]

def preset_choices(dataset=None):
    """Single question list; optional legacy parameter is ignored."""
    return QUESTION_PRESETS


def make_interface(app):
    import gradio as gr
    with gr.Blocks(title="BizAssist AI : Unified Assistant") as ui:
        gr.Markdown("# BizAssist AI : Unified Coffee Shop Assistant")
        gr.Markdown("Choose a **common question** or type your own. BizAssist automatically selects real 2023 sales, synthetic 2026 POS, or the POS knowledge base. Every calculated answer identifies its source. Unclear questions prompt clarification; financial totals are never mixed.")
        with gr.Row():
            with gr.Column():
                preset = gr.Dropdown(choices=preset_choices(), value="", label="Common questions (optional)", interactive=True)
                question = gr.Textbox(label="Business question", lines=3, placeholder="e.g. What were gross sales in March 2023?")
                store = gr.Dropdown(choices=app.stores, value="All stores", label="Store filter (optional, real 2023 data only)")
                with gr.Row():
                    clear = gr.Button("Clear")
                    submit = gr.Button("Submit", variant="primary")
            with gr.Column():
                answer = gr.Textbox(label="BizAssist AI answer", lines=14, interactive=False)
        preset.change(lambda selected: selected or "", inputs=preset, outputs=question)
        submit.click(lambda q,s: app.answer(q,store=s), inputs=[question,store], outputs=answer)
        question.submit(lambda q,s: app.answer(q,store=s), inputs=[question,store], outputs=answer)
        clear.click(lambda: ("", "", "", "All stores"), outputs=[preset,question,answer,store])
    return ui

if __name__ == "__main__":
    app = main(use_embeddings=False)
    make_interface(app).launch(share=False)
