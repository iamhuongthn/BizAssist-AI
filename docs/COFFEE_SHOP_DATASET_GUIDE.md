# Coffee Shop Sales : real-data experiment

**Source:** https://www.kaggle.com/datasets/keremkarayaz/coffee-shop-sales

**Supplied file:** `Coffee Shop Sales.xlsx` inside the user's `archive.zip` (not downloaded or independently verified by this project).

**Observed columns:** transaction_id, transaction_date, transaction_time, transaction_qty, store_id, store_location, product_id, unit_price, product_category, product_type, product_detail.

**Observed file:** 149,116 rows, 80 distinct product IDs, three store locations, Jan 1 to Jun 30 2023. Each transaction_id is unique in this file, so each row is treated as one transaction and one item line; this is a property of the supplied file, not a universal POS rule.

## Design decisions

- Preserve the original synthetic pipeline, where statuses, discounts, and refunds are known.
- Convert the Kaggle file to **separate** `data/real/processed/{orders,order_items,products}.csv` for schema illustration. Never feed these through the synthetic `preprocess.process()` without adapting its financial requirements.
- Source has no explicit currency field; output assumes USD from NYC store locations, and labels the assumption.
- Source has no status field. `completed_assumed` is a proxy label, **not** an observed POS status.
- Report **gross transaction revenue**, not verified net sales. Gross AOV = gross transaction revenue / distinct transaction IDs.
- A time-ordered 75/25 train/test split is used for a *contemporaneous explanatory* Random Forest experiment on daily gross transaction revenue. Same-day counts and units would not be known in advance, so do **not** call it future-sales forecasting.
- The pre-trained Sentence Transformer and the project's existing knowledge base remain unchanged. No LLM or embedding fine-tuning occurs.
- The source file contains no missing values, duplicate transaction IDs, or nonpositive quantities/prices. Thus the raw/clean row count chart shows no change; this is accurate, not a preprocessing failure. The synthetic pipeline's separate controlled dirty-data audit remains available.

## Run

From the project root:

```bash
pip install -r requirements.txt
python run_coffee_shop.py  # TF-IDF fallback, CPU-friendly
pytest -q tests/test_coffee_shop.py
```

In Google Colab, upload the project ZIP, open `notebooks/BizAssist_CoffeeShop_Real_Data.ipynb`, and run cells. The notebook includes optional Sentence Transformers retrieval and Gradio.

## Files produced

- `data/real/processed/coffee_shop_validated.csv`, `coffee_shop_rejected.csv`, `orders.csv`, `order_items.csv`, `products.csv`, `daily_sales_features.csv`, `encoded_features_unscaled.csv`
- `reports/results/coffee_shop_quality.csv`, `coffee_shop_feature_importance.csv`, `coffee_shop_model_metrics.csv`, `coffee_shop_example_interactions.txt`
- `reports/figures/coffee_shop_before_after.png`, `coffee_shop_daily_revenue.png`, `coffee_shop_feature_importance.png`

## Example supported questions

- What were gross sales in March 2023?
- Show top five products by units sold from 2023-03-01 to 2023-03-31
- What was the busiest hour?
- What is gross sales?
- What were net sales in March 2023? (correctly reports unavailable)

## Evaluation limitations

The original 50-question test set is for the synthetic dataset. Its reported numeric scores do not transfer to this real-data experiment. The included tests independently recompute gross revenue from raw quantity and unit price and verify store filters and rejection logic. A separate manually reviewed, real-data question set would be required to claim comprehensive assistant performance on Kaggle data.
