# BizAssist AI | ITAI 2377 Final

**Huong Nguyen | Group 6 (individual)**

BizAssist answers coffee shop business questions from the correct source. It routes requests between 2023 real transaction data, 2026 simulated POS orders, and a POS knowledge base. Financial figures use business rules, not generated guesses.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iamhuongthn/BizAssist-AI/blob/master/notebooks/BizAssist_AI_Final_Group6.ipynb)

**Repository:** https://github.com/iamhuongthn/BizAssist-AI  
**Main notebook:** [`notebooks/BizAssist_AI_Final_Group6.ipynb`](notebooks/BizAssist_AI_Final_Group6.ipynb)

## Run in Colab

1. Open the notebook using the badge above.
2. Select **Runtime > Run all** with a CPU runtime. The first run installs dependencies and downloads the sentence embedding model.
3. Use the Gradio link printed by the interface cell. Try the preset questions.
4. Review the saved metrics, charts and manual evaluation. Export an evidence ZIP only if needed.

The notebook clones the public repository directly. **No manual ZIP upload is required.** A fresh runtime requires internet access to GitHub and model downloads.

## Run locally

```bash
git clone https://github.com/iamhuongthn/BizAssist-AI.git
cd BizAssist-AI
python -m pip install -r requirements.txt
python run_unified.py
python -m pytest -q tests/
```

Local `run_unified.py` uses TF IDF by default for quicker startup. Colab uses sentence embeddings. Python 3.10 or later is recommended.

## Data scope

| Source | Period | Supported operations |
| --- | --- | --- |
| [Kaggle Coffee Shop Sales](https://www.kaggle.com/datasets/keremkarayaz/coffee-shop-sales) | 2023, 149,116 input rows | Gross revenue, product rankings, volume, busiest hours |
| Generated POS orders | 2026, simulated | Gross and net sales, discounts, refunds, average order value |
| Project knowledge base | Definitions | Business terms and calculation rules |

The source datasets stay separate. The real Coffee Shop file does not provide verified refunds, discounts or completed order status, so BizAssist cannot calculate verified real net sales. USD is a location based assumption, not a stated dataset currency.

## Project map

| Path | Purpose |
| --- | --- |
| `notebooks/` | Complete Colab workflow and live demo |
| `run_unified.py` | Automatic source routing and Gradio interface |
| `run.py`, `run_coffee_shop.py` | Dataset specific pipelines |
| `bizassist/` | Cleaning, features, retrieval, calculations and evaluation |
| `data/` | Raw, processed, real and audit datasets |
| `evaluation/`, `tests/` | Manual cases and automated checks |
| `reports/` | Saved figures, metrics and sample answers |
| `docs/` | Dataset notes and demo instructions |

## Results and limits

The saved notebook reports 30 passing fixed manual cases and nine passing automated tests. Generated evaluation cases are not an independent benchmark. Embedding retrieval and TF IDF should be compared without claiming improvement when their results tie. The revenue experiment estimates same day totals using same day inputs and is **not a future sales forecast**.

## Submission and live demo

Submit `FN_HuongNguyen_Group6_ITAI2377.pdf` and include the [viewable Colab notebook](https://colab.research.google.com/github/iamhuongthn/BizAssist-AI/blob/master/notebooks/BizAssist_AI_Final_Group6.ipynb). The source, dataset samples and evaluation artifacts are available in this repository. See [`docs/DEMO_AND_SUBMISSION.md`](docs/DEMO_AND_SUBMISSION.md) for a short demo script. The final PDF is a separate Canvas deliverable.

**Contribution:** Huong Nguyen is the only member of Group 6 and completed preparation, feature engineering, implementation, testing, analysis and documentation.
