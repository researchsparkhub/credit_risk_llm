# credit_risk_llm

Rework of the paper *"Random Forest Structure as Prompt Guidance for Credit Risk
Prediction with Large Language Models"* (Research Spark Hub).

Dataset: **Default of Credit Card Clients** (UCI / Kaggle) — see
`data/DATA_PROVENANCE.md`. Regenerate the CSV with `python convert_xls_to_csv.py`.

## Contents
- `data/UCI_Credit_Card.csv` — the working dataset (Kaggle schema, 30k rows).
- `convert_xls_to_csv.py` — builds the CSV from the canonical UCI `.xls`.
- `credit_risk_interpretability_vs_performance.py` — a 4-week DT/RF vs LogReg/MLP
  teaching Colab (note: contains no LLM prompting; the paper's LLM notebook is
  separate and still to be added).

## Setup
```
python3 -m venv .venv && source .venv/bin/activate
pip install pandas xlrd openpyxl scikit-learn
```
