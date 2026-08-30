"""
Honest rerun of the credit-default LLM experiment.

First pass: N=300 eval cases, core conditions only (no ablation yet):
  - majority baseline
  - Logistic Regression (fitted)
  - Random Forest (fitted)          [ = paper's "Prompt 3" ]
  - L1: direct LLM prediction        [ = paper's "Prompt 1", done for real ]
  - L2: RF-informed LLM prediction   [ = paper's "Prompt 2", done for real ]

LLM calls are batched and cached to results/cache_*.json so re-runs are free.
"""
import os, json, time, hashlib, re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, accuracy_score, roc_auc_score, average_precision_score)
import anthropic

# ---------------- config ----------------
SEED = 42
N_EVAL = 300
K_SHOT = 12                       # balanced few-shot examples (6 default + 6 non-default)
BATCH = 10                        # clients per LLM call
MODEL = "claude-sonnet-5"
RESULTS = "results"; os.makedirs(RESULTS, exist_ok=True)
FEATURES = None                   # set after load

client = anthropic.Anthropic()   # uses ANTHROPIC_API_KEY

# ---------------- data ----------------
df = pd.read_csv("data/UCI_Credit_Card.csv").rename(columns={"default.payment.next.month": "DEFAULT"})
df = df.drop(columns=["ID"])
df["DEFAULT"] = df["DEFAULT"].astype(int)
X = df.drop(columns=["DEFAULT"]); y = df["DEFAULT"]
FEATURES = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)

# fixed, stratified 300-case eval set from the test split
Xe, _, ye, _ = train_test_split(X_test, y_test, train_size=N_EVAL, random_state=SEED, stratify=y_test)
Xe = Xe.reset_index(drop=True); ye = ye.reset_index(drop=True)
actual = ye.tolist()

# ---------------- fitted baselines ----------------
rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=SEED, n_jobs=-1).fit(X_train, y_train)
logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)).fit(X_train, y_train)

rf_pred = rf.predict(Xe); rf_proba = rf.predict_proba(Xe)[:, 1]
lr_pred = logreg.predict(Xe); lr_proba = logreg.predict_proba(Xe)[:, 1]
maj_pred = [0] * len(actual)

importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
top_feats = list(importances.head(8).index)

# ---------------- few-shot pool (from TRAIN) ----------------
rng = np.random.RandomState(SEED)
def pick(cls, k):
    idx = y_train[y_train == cls].index
    return X_train.loc[rng.choice(idx, k, replace=False)]
shots = pd.concat([pick(1, K_SHOT // 2), pick(0, K_SHOT // 2)])
shots_y = y_train.loc[shots.index]

def row_text(r):
    return ", ".join(f"{c}={int(r[c])}" for c in FEATURES)

def shots_block():
    lines = []
    for (_, r), lab in zip(shots.iterrows(), shots_y):
        lines.append(f"- {row_text(r)} -> {int(lab)}")
    return "\n".join(lines)

SHOTS = shots_block()

# ---------------- prompts ----------------
COMMON_TASK = (
    "You are predicting credit-card default (1 = will default next month, 0 = will not) "
    "for clients in the UCI 'Default of Credit Card Clients' dataset.\n"
    "Feature notes: PAY_0..PAY_6 are repayment-status codes (>=1 means months of delay), "
    "BILL_AMT1..6 are recent bill amounts, PAY_AMT1..6 are recent payment amounts, "
    "LIMIT_BAL is the credit limit.\n"
    f"Here are {K_SHOT} labeled examples (features -> label):\n{SHOTS}\n\n"
)

L1_INSTR = "Predict the label for each client below using general repayment-behavior patterns.\n\n"
L2_INSTR = (
    "Act as a trained Random Forest classifier. A fitted Random Forest on this data ranks these "
    f"features as most important (in order): {', '.join(top_feats)}. "
    "Weigh repayment-status (PAY_*) most, then bill/payment amounts, when deciding.\n\n"
)

OUTPUT_RULE = (
    "Return ONLY a JSON array, one object per client, in the same order, like:\n"
    '[{"id": 0, "label": 0, "confidence": 0.0}, ...]\n'
    "confidence = your probability that label=1 (default), between 0 and 1. No prose."
)

def build_prompt(instr, batch_rows, start_id):
    clients = "\n".join(f"id {start_id+j}: {row_text(r)}" for j, (_, r) in enumerate(batch_rows.iterrows()))
    return COMMON_TASK + instr + "Clients to predict:\n" + clients + "\n\n" + OUTPUT_RULE

# ---------------- LLM runner with cache ----------------
def call_llm(prompt):
    kwargs = dict(model=MODEL, max_tokens=2000,
                  messages=[{"role": "user", "content": prompt}])
    try:
        msg = client.messages.create(thinking={"type": "disabled"}, **kwargs)
    except Exception:
        msg = client.messages.create(**kwargs)   # if thinking can't be disabled, fall back
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            return block.text
    raise RuntimeError("no text block in response")

def parse(txt, n, start_id):
    m = re.search(r"\[.*\]", txt, re.S)
    arr = json.loads(m.group(0))
    out = {}
    for o in arr:
        out[int(o["id"])] = (int(o["label"]), float(o.get("confidence", o["label"])))
    return [out.get(start_id + j, (0, 0.0)) for j in range(n)]

def run_condition(name, instr):
    cache_path = f"{RESULTS}/cache_{name}.json"
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    preds, confs = [], []
    t0 = time.time()
    for s in range(0, len(Xe), BATCH):
        batch = Xe.iloc[s:s+BATCH]
        key = hashlib.md5((name + str(s) + build_prompt(instr, batch, s)).encode()).hexdigest()
        if key in cache:
            res = cache[key]
        else:
            txt = call_llm(build_prompt(instr, batch, s))
            res = parse(txt, len(batch), s)
            cache[key] = res
            json.dump(cache, open(cache_path, "w"))
            print(f"  {name}: {s+len(batch)}/{len(Xe)}")
        for lab, cf in res:
            preds.append(int(lab)); confs.append(float(cf))
    dt = (time.time() - t0) / len(Xe)
    return preds, confs, dt

# ---------------- metrics ----------------
def metrics(name, pred, proba=None, sec=None):
    tn, fp, fn, tp = confusion_matrix(actual, pred, labels=[0, 1]).ravel()
    row = dict(condition=name,
               accuracy=round(100*accuracy_score(actual, pred), 2),
               precision=round(precision_score(actual, pred, zero_division=0), 4),
               recall=round(recall_score(actual, pred, zero_division=0), 4),
               f1=round(f1_score(actual, pred, zero_division=0), 4),
               pred_default_pct=round(100*sum(pred)/len(pred), 1),
               TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))
    row["auc_roc"] = round(roc_auc_score(actual, proba), 4) if proba is not None and len(set(proba)) > 1 else None
    row["auc_pr"] = round(average_precision_score(actual, proba), 4) if proba is not None and len(set(proba)) > 1 else None
    row["sec_per_client"] = round(sec, 4) if sec else None
    return row

def main():
    print(f"eval N={len(actual)}  default rate={100*sum(actual)/len(actual):.1f}%")
    print("RF top features:", top_feats, "\n")
    rows = [
        metrics("Majority (no default)", maj_pred),
        metrics("Logistic Regression", lr_pred, lr_proba),
        metrics("Random Forest", rf_pred, rf_proba),
    ]
    print("running L1 (direct LLM)...");     p1, c1, t1 = run_condition("L1_direct", L1_INSTR)
    print("running L2 (RF-informed LLM)..."); p2, c2, t2 = run_condition("L2_rf_informed", L2_INSTR)
    rows.append(metrics("L1: Direct LLM", p1, c1, t1))
    rows.append(metrics("L2: RF-informed LLM", p2, c2, t2))
    res = pd.DataFrame(rows)
    res.to_csv(f"{RESULTS}/results_N{N_EVAL}.csv", index=False)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\n================ RESULTS (N=%d) ================" % N_EVAL)
    print(res.to_string(index=False))
    print(f"\nsaved -> {RESULTS}/results_N{N_EVAL}.csv")
    print("model:", MODEL, "| few-shot k:", K_SHOT, "| batch:", BATCH)

if __name__ == "__main__":
    main()
