# credit_risk_llm

Honest re-study and rewritten paper for **credit-card default prediction**, examining
**when a fitted classifier helps a large language model (LLM) — and vice versa** — on the
UCI *Default of Credit Card Clients* dataset.

Paper: `paper/main.pdf` (source `paper/main.tex`, refs `paper/references.bib`).

> **Note on scope.** An earlier draft's "LLM" conditions were, on inspection of the code,
> arithmetic heuristics (`row.sum()` / `row.mean()`), not LLM calls. This repository reruns
> everything with a **real LLM** (`claude-sonnet-5`) and fitted baselines, with
> imbalance-aware metrics and bootstrap confidence intervals. `repro_paper.py` reproduces the
> original heuristic numbers for the record.

## Key findings
- A few-shot LLM is **recall/F1-competitive** with fitted models but **weaker at ranking (AUC)**.
- **Imitation prompting** ("act as a Random Forest" / feature-list hints) gives **no** significant gain.
- **Classifier-guided feature pruning** (show only the RF top-8 features) **significantly helps**; **few-shot example curation hurts**.
- **Hybrid RF→LLM** (feed the RF's probability into the prompt) is best: **RF-level AUC + higher recall**. The reverse (LLM→RF) and multi-classifier inputs do **not** help.
- **Diffusion** minority augmentation beats **SMOTE** (AUC-PR/precision) but not simple **class-weighting**.

---

## 1. Setup
Requires Python 3.11+ and (for the LLM experiments) an Anthropic API key.

```bash
cd credit_risk_llm
python3 -m venv .venv && source .venv/bin/activate
pip install pandas xlrd openpyxl scikit-learn imbalanced-learn scipy torch anthropic matplotlib
export ANTHROPIC_API_KEY=sk-...        # only needed to (re)run LLM conditions
```
LLM responses are **cached** under `results/cache_*.json`, so re-running the LLM scripts
with the caches present makes **no** API calls and reproduces the exact numbers.

## 2. Data
```bash
python convert_xls_to_csv.py           # UCI .xls -> data/UCI_Credit_Card.csv (Kaggle schema)
```
The raw `.xls` (gitignored, 5.3 MB) is downloaded from UCI; provenance + md5 in
`data/DATA_PROVENANCE.md`. Dataset: 30,000 clients, 22.1% default. Split: stratified 80/20,
seed 42. LLM conditions evaluate on a fixed **stratified 2,000-client** subset of the test set.

## 3. Reproduce the experiments
| Script | What it produces | LLM calls? |
|---|---|---|
| `repro_paper.py` | reproduces the original heuristic numbers (Prompt 1/2 = `row.sum`/`row.mean`) | no |
| `run_experiment.py` | first-pass N=300: baselines + L1/L2 | yes (cached) |
| `run_ablation.py` | **N=2000** imitation ablation: L1, L2, L2a, L2b + baselines + bootstrap CIs | yes (cached) |
| `run_l2b_ext.py` | classifier-guided prompt construction: E1–E5 (feature pruning; boundary/representative shots) | yes (cached) |
| `run_hybrid.py` | **E6** hybrid (RF probability into the prompt) | yes (cached) |
| `run_ensemble_hybrid.py` | **E7** (RF+LogReg+AdaBoost into the prompt) | yes (cached) |
| `run_inverse.py` | inverse hybrid (LLM output as an RF feature; 5-fold CV) | no |
| `diffusion_aug.py` / `diffusion_aug_v2.py` / `diffusion_aug_tabddpm.py` | diffusion minority augmentation (v2 is the kept generator) | no |
| `diffusion_cis.py` | 5-seed CIs for the diffusion-vs-SMOTE comparison | no |
| `make_figures.py` | `paper/fig_tradeoff.pdf`, `paper/fig_roc_pr.pdf` | no |

Typical order:
```bash
source .venv/bin/activate
python run_ablation.py          # writes results/ablation_N2000.csv (+ caches)
python run_l2b_ext.py           # E1–E5   -> results/l2b_ext_N2000.csv
python run_hybrid.py            # E6      -> results/hybrid_N2000.csv
python run_ensemble_hybrid.py   # E7      -> results/ensemble_hybrid_N2000.csv
python run_inverse.py           # inverse -> results/inverse_stack_N2000.csv
python diffusion_cis.py         # diffusion CIs -> results/diffusion_cis_summary.csv
python make_figures.py          # figures
```
Note: `run_l2b_ext.py`, `run_hybrid.py`, `run_ensemble_hybrid.py`, and `run_inverse.py`
import `run_ablation.py` (shared split/eval/plumbing), so its cache must exist first.

## 4. Results & outputs
All metric tables are in `results/*.csv`; cached LLM outputs in `results/cache_*.json`.
The paper's tables/figures are generated from these files.

## 5. Build the paper
```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## 6. Configuration
- LLM: `claude-sonnet-5`, extended thinking disabled, 12-shot balanced examples, batched 20/call.
- Fitted models: Random Forest (100 trees, depth 12), logistic regression (standardized), AdaBoost (200).
- Uncertainty: 1,000 bootstrap resamples (paired for between-condition tests); diffusion over 5 seeds.
- Seeds fixed at 42 (diffusion CIs use seeds 0–4).

## 7. Caveats
Single LLM, single dataset, one 2,000-client eval draw, one few-shot draw per condition.
Bootstrap CIs cover test-set sampling, not LLM/prompt-draw variance. See the paper's
Limitations section. Reference author fields in `references.bib` were sourced from the
papers' arXiv/venue pages; verify before camera-ready submission.
