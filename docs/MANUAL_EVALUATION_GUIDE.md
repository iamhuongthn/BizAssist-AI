# Independent evaluation extension : 30 fixed cases

- `evaluation/real_coffee_shop_manual_test.csv`: 15 fixed questions covering months, dates, stores, rankings, and unsupported metrics.
- `evaluation/synthetic_manual_test.csv`: 15 fixed questions covering financial arithmetic, rankings, curated FAQ citations, and unsupported requests.
- `bizassist/manual_evaluation.py`: derives gold totals from raw Kaggle XLSX and raw synthetic orders/items CSV, without using the assistant calculation helpers.
- `reports/results/manual_evaluation_results_30.csv`: every expected value, actual answer, pass/fail, and latency.
- `reports/results/manual_evaluation_summary.csv`: pass rates by dataset and type.

Run the master notebook, or after `app = main(...)`, call `evaluate_manual(app)`. Do not modify the fixed questions after reviewing failures; retain failures in the final report.

These cases were manually authored for the project, not independently audited by an external reviewer. Gold financial values use independent aggregation code but the same underlying raw source data. This is separate from the 50 automatically generated synthetic evaluation questions.
