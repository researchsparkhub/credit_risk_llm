"""
LLM ablation at N=2000 with bootstrap CIs.

LLM conditions (isolate role vs. feature-emphasis):
  L1   direct (no guidance)
  L2   role + feature-emphasis  (full RF-informed)
  L2a  role only  ("act as a Random Forest", no feature list)
  L2b  feature-emphasis only    (weigh PAY_*/bills, no RF role)
Fitted baselines (majority, LogReg, RF) on the same 2000 eval cases.
Bootstrap 95% CIs per condition + paired (Lx - L1) differences.
Thinking disabled (near-deterministic); calls batched + cached.
"""
import os, json, time, hashlib, re
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, accuracy_score, roc_auc_score, average_precision_score)
import anthropic

SEED = 42; N_EVAL = 2000; K_SHOT = 12; BATCH = 20
MODEL = "claude-sonnet-5"; RESULTS = "results"; os.makedirs(RESULTS, exist_ok=True)
rng = np.random.RandomState(SEED)
client = anthropic.Anthropic()

df = pd.read_csv("data/UCI_Credit_Card.csv").rename(columns={"default.payment.next.month": "DEFAULT"}).drop(columns=["ID"])
df["DEFAULT"] = df["DEFAULT"].astype(int)
X = df.drop(columns=["DEFAULT"]); y = df["DEFAULT"]; FEATURES = list(X.columns)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
Xe, _, ye, _ = train_test_split(Xte, yte, train_size=N_EVAL, random_state=SEED, stratify=yte)
Xe = Xe.reset_index(drop=True); ye = ye.reset_index(drop=True); actual = ye.tolist()

rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)).fit(Xtr, ytr)
top_feats = list(pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False).head(8).index)

def pick(cls, k):
    idx = ytr[ytr == cls].index
    return Xtr.loc[rng.choice(idx, k, replace=False)]
shots = pd.concat([pick(1, K_SHOT // 2), pick(0, K_SHOT // 2)]); shots_y = ytr.loc[shots.index]
def row_text(r): return ", ".join(f"{c}={int(r[c])}" for c in FEATURES)
SHOTS = "\n".join(f"- {row_text(r)} -> {int(l)}" for (_, r), l in zip(shots.iterrows(), shots_y))

COMMON = ("You are predicting credit-card default (1 = will default next month, 0 = will not) "
          "for UCI 'Default of Credit Card Clients' data.\n"
          "PAY_0..PAY_6 = repayment-status codes (>=1 means months delayed), BILL_AMT* = bills, "
          "PAY_AMT* = payments, LIMIT_BAL = credit limit.\n"
          f"{K_SHOT} labeled examples (features -> label):\n{SHOTS}\n\n")
TOP = ", ".join(top_feats)
INSTR = {
    "L1_direct":       "Predict each client's label using general repayment-behavior patterns.\n\n",
    "L2_full":         f"Act as a trained Random Forest classifier. A fitted Random Forest ranks these features most important (in order): {TOP}. Weigh repayment-status (PAY_*) most, then bill/payment amounts.\n\n",
    "L2a_role_only":   "Act as a trained Random Forest classifier. Use your own judgment about which features matter most.\n\n",
    "L2b_feats_only":  f"When deciding, weigh these features most (in order): {TOP}. Weigh repayment-status (PAY_*) most, then bill/payment amounts.\n\n",
}
OUTPUT = ('Return ONLY a JSON array, one object per client in order:\n'
          '[{"id":0,"label":0,"confidence":0.0}, ...]\n'
          "confidence = probability that label=1 (default), 0..1. No prose.")

def build(instr, batch, start):
    clients = "\n".join(f"id {start+j}: {row_text(r)}" for j, (_, r) in enumerate(batch.iterrows()))
    return COMMON + instr + "Clients:\n" + clients + "\n\n" + OUTPUT

def call_llm(prompt):
    kw = dict(model=MODEL, max_tokens=3000, messages=[{"role": "user", "content": prompt}])
    try: msg = client.messages.create(thinking={"type": "disabled"}, **kw)
    except Exception: msg = client.messages.create(**kw)
    for b in msg.content:
        if getattr(b, "type", None) == "text": return b.text
    raise RuntimeError("no text block")

def parse(txt, n, start):
    arr = json.loads(re.search(r"\[.*\]", txt, re.S).group(0))
    d = {int(o["id"]): (int(o["label"]), float(o.get("confidence", o["label"]))) for o in arr}
    return [d.get(start + j, (0, 0.0)) for j in range(n)]

def run_condition(name):
    cache_path = f"{RESULTS}/cache_abl_{name}.json"
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    preds, confs = [], []
    for s in range(0, len(Xe), BATCH):
        batch = Xe.iloc[s:s+BATCH]
        key = hashlib.md5((name + str(s) + build(INSTR[name], batch, s)).encode()).hexdigest()
        if key not in cache:
            cache[key] = parse(call_llm(build(INSTR[name], batch, s)), len(batch), s)
            json.dump(cache, open(cache_path, "w"))
            print(f"  {name}: {s+len(batch)}/{len(Xe)}", flush=True)
        for lab, cf in cache[key]:
            preds.append(int(lab)); confs.append(float(cf))
    return np.array(preds), np.array(confs)

# ---- metrics + bootstrap ----
A = np.array(actual)
def mvec(pred, proba=None):
    tn, fp, fn, tp = confusion_matrix(A, pred, labels=[0, 1]).ravel()
    d = dict(accuracy=accuracy_score(A, pred), precision=precision_score(A, pred, zero_division=0),
             recall=recall_score(A, pred, zero_division=0), f1=f1_score(A, pred, zero_division=0),
             pred_def=pred.mean(), TP=int(tp), FN=int(fn), FP=int(fp))
    d["auc_roc"] = roc_auc_score(A, proba) if proba is not None and len(set(proba)) > 1 else np.nan
    d["auc_pr"] = average_precision_score(A, proba) if proba is not None and len(set(proba)) > 1 else np.nan
    return d

B = 1000
boot_idx = [rng.randint(0, len(A), len(A)) for _ in range(B)]
def boot_ci(pred, proba, metric):
    vals = []
    for idx in boot_idx:
        aa = A[idx]
        if aa.sum() == 0 or aa.sum() == len(aa):  # skip degenerate resamples for auc/recall
            continue
        pp = pred[idx]
        if metric == "f1": vals.append(f1_score(aa, pp, zero_division=0))
        elif metric == "recall": vals.append(recall_score(aa, pp, zero_division=0))
        elif metric == "accuracy": vals.append(accuracy_score(aa, pp))
        elif metric == "auc_roc": vals.append(roc_auc_score(aa, proba[idx]))
        elif metric == "auc_pr": vals.append(average_precision_score(aa, proba[idx]))
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)

def paired(predA, probA, predB, probB, metric):
    diffs = []
    for idx in boot_idx:
        aa = A[idx]
        if aa.sum() == 0 or aa.sum() == len(aa): continue
        if metric == "auc_roc":
            diffs.append(roc_auc_score(aa, probA[idx]) - roc_auc_score(aa, probB[idx]))
        elif metric == "f1":
            diffs.append(f1_score(aa, predA[idx], zero_division=0) - f1_score(aa, predB[idx], zero_division=0))
        elif metric == "recall":
            diffs.append(recall_score(aa, predA[idx], zero_division=0) - recall_score(aa, predB[idx], zero_division=0))
    lo, hi = np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
    return np.mean(diffs), lo, hi

def main():
    print(f"eval N={len(A)} default={100*A.mean():.1f}% | model={MODEL} | k={K_SHOT} | RF top: {top_feats}\n")
    fitted = {
        "Majority": (np.zeros(len(A), int), None),
        "LogReg": (logreg.predict(Xe), logreg.predict_proba(Xe)[:, 1]),
        "RandomForest": (rf.predict(Xe), rf.predict_proba(Xe)[:, 1]),
    }
    llm = {}
    for name in INSTR:
        print(f"running {name} ...")
        llm[name] = run_condition(name)

    rows = []
    for nm, (p, pr) in fitted.items():
        rows.append({"condition": nm, **{k: round(v, 4) if isinstance(v, float) else v for k, v in mvec(p, pr).items()}})
    for nm, (p, pr) in llm.items():
        rows.append({"condition": nm, **{k: round(v, 4) if isinstance(v, float) else v for k, v in mvec(p, pr).items()}})
    res = pd.DataFrame(rows)
    res.to_csv(f"{RESULTS}/ablation_N{N_EVAL}.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(f"\n============ ABLATION RESULTS (N={N_EVAL}) ============")
    print(res.to_string(index=False))

    print("\n---- bootstrap 95% CIs (LLM conditions) ----")
    for nm, (p, pr) in llm.items():
        f = boot_ci(p, pr, "f1"); a = boot_ci(p, pr, "auc_roc"); r = boot_ci(p, pr, "recall")
        print(f"  {nm:16s} F1 [{f[0]:.3f},{f[1]:.3f}]  AUC-ROC [{a[0]:.3f},{a[1]:.3f}]  recall [{r[0]:.3f},{r[1]:.3f}]")

    print("\n---- paired differences vs L1 (mean [95% CI]); CI excludes 0 = significant ----")
    p1, pr1 = llm["L1_direct"]
    for nm in ["L2_full", "L2a_role_only", "L2b_feats_only"]:
        p2, pr2 = llm[nm]
        for metric in ["f1", "auc_roc", "recall"]:
            m, lo, hi = paired(p2, pr2, p1, pr1, metric)
            sig = "SIG" if lo * hi > 0 else "ns"
            print(f"  {nm:16s} {metric:8s} {m:+.4f} [{lo:+.4f},{hi:+.4f}] {sig}")
    print(f"\nsaved -> {RESULTS}/ablation_N{N_EVAL}.csv")

if __name__ == "__main__":
    main()
