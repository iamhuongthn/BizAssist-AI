# BizAssist AI | ITAI 2377 Final Project

**Author:** Huong Nguyen  
**Group:** 6, individual project  
**Deliverables:** Colab notebook, final PDF report, source code and evaluation evidence

## Purpose
BizAssist answers coffee shop business questions using validated records. It routes each question to the right source, calculates financial results with explicit rules, and explains when a metric cannot be verified.

## Start in Google Colab
1. Upload `BizAssist_AI_Final_Group6.zip` to the Colab Files panel.
2. Open `BizAssist_AI_Final_Group6.ipynb` in Colab.
3. Select **Runtime > Run all**. A CPU runtime is sufficient. The first run downloads the embedding model.
4. Open the Gradio interface. Choose a preset question or enter your own.
5. Run the evaluation and export cells. Download generated outputs before the session ends.

The notebook extracts the ZIP and runs the full pipeline. Do not upload the separate outputs ZIP as the source archive.

## Run locally
```bash
pip install -r requirements.txt
python run_unified.py
python -m pytest -q tests/
```
The local entry point uses TF IDF for faster startup. The Colab notebook uses sentence embeddings.

## Data and boundaries
| Source | Scope | Supported answers |
| --- | --- | --- |
| Kaggle Coffee Shop Sales | 2023, 149,116 source rows | Gross sales, product rankings, transaction volume, busiest hour |
| Generated POS transactions | 2026, simulated | Gross sales, net sales, discounts, refunds, average order value |
| Project knowledge base | Curated definitions | POS terms and calculation definitions |

**Financial integrity:** The Kaggle dataset does not contain verified discount, refund, or completed order status fields. Do not claim real net sales from it. USD is assumed from the New York locations, not specified in the source file. The two datasets are never added together.

Kaggle source: https://www.kaggle.com/datasets/keremkarayaz/coffee-shop-sales

## Main files
| File or folder | Purpose |
| --- | --- |
| `BizAssist_AI_Final_Group6.ipynb` | Main Colab workflow, evaluation and demo |
| `run_unified.py` | Query routing and Gradio interface |
| `run.py` | Synthetic POS pipeline |
| `run_coffee_shop.py` | Real Coffee Shop pipeline |
| `bizassist/` | Preprocessing, engineered features, retrieval, business rules and evaluation |
| `data/` | Raw, processed and audit data |
| `evaluation/` | Fixed manual evaluation questions |
| `reports/results/` | Saved metrics and example interactions |
| `reports/figures/` | Data quality and feature importance charts |
| `tests/` | Automated tests |
| `docs/` | Dataset and evaluation notes |

## Evaluation notes
The notebook reports 50 generated synthetic cases and 30 fixed manual cases. Generated cases are not an independent benchmark. Retrieval results must be compared with the TF IDF baseline without claiming an improvement when scores tie. The real data revenue model estimates same day totals using same day inputs. It is not a future sales forecast.

## Submission and live demo
Submit the PDF report and share a viewable Colab link. Keep the source ZIP and evaluation outputs as supporting evidence. For the live demo, show one real gross sales answer, one synthetic net sales answer, and one unsupported real net sales request. The last example demonstrates safe handling of unavailable fields.

**Contribution:** Huong Nguyen completed data preparation, feature engineering, implementation, evaluation, documentation and demonstration as the sole member of Group 6.
