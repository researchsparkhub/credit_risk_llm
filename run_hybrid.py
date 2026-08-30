"""
E6 (hybrid): E1 prompt (top-8 features + random shots) PLUS the Random Forest's
decision as an extra input feature (RF_prob + RF_pred) for every client and shot.
Shots carry OOB RF verdicts (non-overfit); test clients carry the fitted RF's
probability. Tests whether feeding RF's output lifts AUC while keeping LLM recall.
"""
import os, json, hashlib
import numpy as np, pandas as pd
import run_ablation as R
import run_l2b_ext as X          # top_feats, rand_shots/rand_y, oob_p, rf_oob

BATCH = R.BATCH; RESULTS = R.RESULTS
top_feats = X.top_feats
sh, shy = X.rand_shots, X.rand_y

# RF verdicts: OOB for shots (honest), fitted-RF for test clients
pos = {lab: i for i, lab in enumerate(R.Xtr.index)}
sh_prob = np.array([X.oob_p[pos[lab]] for lab in sh.index])
sh_pred = (sh_prob >= 0.5).astype(int)
te_prob = R.rf.predict_proba(R.Xe)[:, 1]
te_pred = (te_prob >= 0.5).astype(int)

def rowtxt(r, feats): return ", ".join(f"{c}={int(r[c])}" for c in feats)
PRE = ("You are predicting credit-card default (1 = will default next month, 0 = will not) "
       "for UCI 'Default of Credit Card Clients' data.\n"
       "PAY_0..PAY_6 = repayment-status codes (>=1 = months delayed), BILL_AMT*=bills, PAY_AMT*=payments, LIMIT_BAL=credit limit.\n"
       "Each row also includes RF_prob (a trained Random Forest's estimated probability of default) "
       "and RF_pred (its 0/1 decision). Combine RF's opinion with the features to decide.\n")
INSTR = "Predict each client's label.\n\n"
OUTPUT = ('Return ONLY a JSON array, one object per client in order:\n'
          '[{"id":0,"label":0,"confidence":0.0}, ...]\n'
          "confidence = probability that label=1 (default), 0..1. No prose.")

def shots_block():
    return "\n".join(f"- {rowtxt(r, top_feats)}, RF_prob={sh_prob[i]:.2f}, RF_pred={sh_pred[i]} -> {int(l)}"
                     for i, ((_, r), l) in enumerate(zip(sh.iterrows(), shy)))
SB = shots_block()

def build(batch, start):
    clients = "\n".join(
        f"id {start+j}: {rowtxt(r, top_feats)}, RF_prob={te_prob[start+j]:.2f}, RF_pred={te_pred[start+j]}"
        for j, (_, r) in enumerate(batch.iterrows()))
    return PRE + f"{len(sh)} labeled examples:\n{SB}\n\n" + INSTR + "Clients:\n" + clients + "\n\n" + OUTPUT

def run():
    cache_path = f"{RESULTS}/cache_hyb_E6.json"
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    preds, confs = [], []
    for s in range(0, len(R.Xe), BATCH):
        batch = R.Xe.iloc[s:s+BATCH]; prompt = build(batch, s)
        key = hashlib.md5(("E6" + str(s) + prompt).encode()).hexdigest()
        if key not in cache:
            cache[key] = R.parse(R.call_llm(prompt), len(batch), s)
            json.dump(cache, open(cache_path, "w"))
            print(f"  E6: {s+len(batch)}/{len(R.Xe)}", flush=True)
        for lab, cf in cache[key]:
            preds.append(int(lab)); confs.append(float(cf))
    return np.array(preds), np.array(confs)

def main():
    print("sample RF_prob for first 5 test clients:", np.round(te_prob[:5], 2).tolist())
    print("running E6 (hybrid: E1 + RF verdict) ...")
    p6, c6 = run()
    p1, pr1 = R.run_condition("L1_direct")
    pe1, ce1 = X.run_condition("E1_topk")
    rf_p, rf_pr = R.rf.predict(R.Xe), R.rf.predict_proba(R.Xe)[:, 1]

    rows = []
    for nm, (p, pr) in [("L1_direct", (p1, pr1)), ("E1_topk", (pe1, ce1)),
                        ("E6_hybrid(E1+RF)", (p6, c6)), ("RandomForest", (rf_p, rf_pr))]:
        m = R.mvec(p, pr)
        rows.append({"condition": nm, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}})
    res = pd.DataFrame(rows)
    res.to_csv(f"{RESULTS}/hybrid_N{len(R.A)}.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(f"\n============ HYBRID E6 (N={len(R.A)}) ============")
    print(res.to_string(index=False))

    print("\n---- paired diffs (mean [95% CI]); CI excludes 0 = significant ----")
    for base_name, (pb, prb) in [("vs L1", (p1, pr1)), ("vs E1", (pe1, ce1)), ("vs RF", (rf_p, rf_pr))]:
        for metric in ["f1", "auc_roc", "recall"]:
            m, lo, hi = R.paired(p6, c6, pb, prb, metric)
            print(f"  E6 {base_name:6s} {metric:8s} {m:+.4f} [{lo:+.4f},{hi:+.4f}] {'SIG' if lo*hi>0 else 'ns'}")
    print(f"\nsaved -> {RESULTS}/hybrid_N{len(R.A)}.csv")

if __name__ == "__main__":
    main()
