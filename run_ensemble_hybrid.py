"""
E7 (ensemble hybrid): top-8 features + THREE classifier outputs as LLM inputs:
  RF_prob/pred, LR_prob/pred (logistic regression), ADA_prob/pred (AdaBoost).
Shots carry out-of-fold verdicts (honest); test clients carry fitted-model probs.
Compares E7 vs E6 (RF-only hybrid), E1 (features only), and the fitted baselines.
"""
import os, json, hashlib
import numpy as np, pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.base import clone
import run_ablation as R
import run_l2b_ext as X
import run_hybrid as H            # for cached E6 preds

BATCH = R.BATCH; top_feats = X.top_feats
sh, shy = X.rand_shots, X.rand_y

# ---- fit AdaBoost on 24k train; RF + LogReg already fitted in R ----
ada = AdaBoostClassifier(n_estimators=200, random_state=R.SEED).fit(R.Xtr, R.ytr)

# ---- test-client probabilities (honest: models fit on train, applied to held-out) ----
rf_te = R.rf.predict_proba(R.Xe)[:, 1]
lr_te = R.logreg.predict_proba(R.Xe)[:, 1]
ada_te = ada.predict_proba(R.Xe)[:, 1]

# ---- out-of-fold probabilities for shot rows (honest) ----
pos = {lab: i for i, lab in enumerate(R.Xtr.index)}
lr_oof = cross_val_predict(clone(R.logreg), R.Xtr, R.ytr, cv=5, method="predict_proba", n_jobs=-1)[:, 1]
ada_oof = cross_val_predict(AdaBoostClassifier(n_estimators=200, random_state=R.SEED),
                            R.Xtr, R.ytr, cv=5, method="predict_proba", n_jobs=-1)[:, 1]
def shot_probs(lab):
    i = pos[lab]
    return X.oob_p[i], lr_oof[i], ada_oof[i]

def rowtxt(r, feats): return ", ".join(f"{c}={int(r[c])}" for c in feats)
def verdict(rf_p, lr_p, ada_p):
    return (f"RF_prob={rf_p:.2f}, RF_pred={int(rf_p>=0.5)}, "
            f"LR_prob={lr_p:.2f}, LR_pred={int(lr_p>=0.5)}, "
            f"ADA_prob={ada_p:.2f}, ADA_pred={int(ada_p>=0.5)}")

PRE = ("You are predicting credit-card default (1=will default next month, 0=will not) for UCI "
       "'Default of Credit Card Clients' data.\nPAY_0..PAY_6=repayment-status codes (>=1=months delayed), "
       "BILL_AMT*=bills, PAY_AMT*=payments, LIMIT_BAL=credit limit.\n"
       "Each row also includes the estimated default probability and 0/1 decision from THREE trained "
       "classifiers: RF (Random Forest), LR (Logistic Regression), ADA (AdaBoost). "
       "Weigh these model opinions together with the features.\n")
INSTR = "Predict each client's label.\n\n"
OUTPUT = ('Return ONLY a JSON array, one object per client in order:\n'
          '[{"id":0,"label":0,"confidence":0.0}, ...]\nconfidence = P(label=1), 0..1. No prose.')

SB = "\n".join(f"- {rowtxt(r, top_feats)}, {verdict(*shot_probs(idx))} -> {int(l)}"
               for (idx, r), l in zip(sh.iterrows(), shy))

def build(batch, start):
    clients = "\n".join(f"id {start+j}: {rowtxt(r, top_feats)}, {verdict(rf_te[start+j], lr_te[start+j], ada_te[start+j])}"
                        for j, (_, r) in enumerate(batch.iterrows()))
    return PRE + f"{len(sh)} labeled examples:\n{SB}\n\n" + INSTR + "Clients:\n" + clients + "\n\n" + OUTPUT

def run():
    cache_path = f"{R.RESULTS}/cache_hyb_E7.json"
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    preds, confs = [], []
    for s in range(0, len(R.Xe), BATCH):
        batch = R.Xe.iloc[s:s+BATCH]; prompt = build(batch, s)
        key = hashlib.md5(("E7" + str(s) + prompt).encode()).hexdigest()
        if key not in cache:
            cache[key] = R.parse(R.call_llm(prompt), len(batch), s)
            json.dump(cache, open(cache_path, "w"))
            print(f"  E7: {s+len(batch)}/{len(R.Xe)}", flush=True)
        for lab, cf in cache[key]:
            preds.append(int(lab)); confs.append(float(cf))
    return np.array(preds), np.array(confs)

def main():
    print("AdaBoost standalone AUC on 2000:", round(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(R.A, ada_te), 4))
    print("running E7 (ensemble hybrid: top8 + RF+LR+ADA) ...")
    p7, c7 = run()
    p6, c6 = H.run()                         # cached E6
    pe1, ce1 = X.run_condition("E1_topk")    # cached E1

    refs = {
        "E1_topk (feats only)":  (pe1, ce1),
        "E6_hybrid (E1+RF)":     (p6, c6),
        "E7_hybrid (E1+RF+LR+ADA)": (p7, c7),
        "RandomForest":          (R.rf.predict(R.Xe), rf_te),
        "LogReg":                (R.logreg.predict(R.Xe), lr_te),
        "AdaBoost":              (ada.predict(R.Xe), ada_te),
    }
    rows = [{"condition": nm, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in R.mvec(np.asarray(p), np.asarray(pr)).items()}}
            for nm, (p, pr) in refs.items()]
    res = pd.DataFrame(rows)
    res.to_csv(f"{R.RESULTS}/ensemble_hybrid_N{len(R.A)}.csv", index=False)
    pd.set_option("display.width", 230, "display.max_columns", 20)
    print(f"\n============ ENSEMBLE HYBRID E7 (N={len(R.A)}) ============")
    print(res.to_string(index=False))

    print("\n---- paired diffs (mean [95% CI]); CI excludes 0 = significant ----")
    for base, (pb, prb) in [("vs E6", (p6, c6)), ("vs RF", (R.rf.predict(R.Xe), rf_te))]:
        for metric in ["f1", "auc_roc", "recall"]:
            m, lo, hi = R.paired(p7, c7, pb, prb, metric)
            print(f"  E7 {base:6s} {metric:8s} {m:+.4f} [{lo:+.4f},{hi:+.4f}] {'SIG' if lo*hi>0 else 'ns'}")
    print(f"\nsaved -> {R.RESULTS}/ensemble_hybrid_N{len(R.A)}.csv")

if __name__ == "__main__":
    main()
