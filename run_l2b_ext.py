"""
L2b extension: use RF to CONSTRUCT the prompt (not just describe it).
Levers, each vs L1 (all features + random shots), instruction text held constant:
  E1  top-K features only (drop the rest), random shots
  E2  RF-boundary few-shot (12 hardest cases near decision boundary, balanced), all features
  E3  combined: top-K features + boundary shots
Boundary selection uses OUT-OF-BAG RF probabilities (not overfit in-sample).
Reuses the N=2000 eval set, LLM plumbing, and bootstrap infra from run_ablation.
"""
import os, json, hashlib, re
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
import run_ablation as R          # reuses data, split, Xe/ye/A, rf, call_llm, parse, mvec, boot_ci, paired, boot_idx

SEED = R.SEED; BATCH = R.BATCH; K_SHOT = R.K_SHOT; RESULTS = R.RESULTS
TOPK = 8
top_feats = R.top_feats[:TOPK]
FEATURES = R.FEATURES

# ---- shot sets ----
# random (same balanced set used by L1/L2 in run_ablation)
rand_shots, rand_y = R.shots, R.shots_y

# boundary shots via OOB probabilities (honest, non-overfit)
rf_oob = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=SEED,
                                n_jobs=-1, oob_score=True, bootstrap=True).fit(R.Xtr, R.ytr)
oob_p = rf_oob.oob_decision_function_[:, 1]
dist = np.abs(oob_p - 0.5)
half = K_SHOT // 2
def nearest_boundary(cls):
    idx = np.where(R.ytr.values == cls)[0]
    order = idx[np.argsort(dist[idx])][:half]      # closest to 0.5
    return order
bidx = np.concatenate([nearest_boundary(1), nearest_boundary(0)])
bnd_shots = R.Xtr.iloc[bidx]; bnd_y = R.ytr.iloc[bidx]
print("boundary-shot OOB probs (should cluster near 0.5):", np.round(oob_p[bidx], 2).tolist())

# representative shots: most CONFIDENTLY-correct examples (prototypes)
def most_confident(cls, want_high):
    idx = np.where(R.ytr.values == cls)[0]
    order = idx[np.argsort(-oob_p[idx])] if want_high else idx[np.argsort(oob_p[idx])]
    return order[:half]
ridx = np.concatenate([most_confident(1, True), most_confident(0, False)])  # clear defaulters + clear non-defaulters
rep_shots = R.Xtr.iloc[ridx]; rep_y = R.ytr.iloc[ridx]
print("representative-shot OOB probs (defaulters~1, non~0):", np.round(oob_p[ridx], 2).tolist())

# ---- prompt building ----
def row_text(r, feats): return ", ".join(f"{c}={int(r[c])}" for c in feats)
def shots_block(sh, shy, feats):
    return "\n".join(f"- {row_text(r, feats)} -> {int(l)}" for (_, r), l in zip(sh.iterrows(), shy))
PRE = ("You are predicting credit-card default (1 = will default next month, 0 = will not) "
       "for UCI 'Default of Credit Card Clients' data.\n"
       "PAY_0..PAY_6 = repayment-status codes (>=1 means months delayed), BILL_AMT* = bills, "
       "PAY_AMT* = payments, LIMIT_BAL = credit limit.\n")
INSTR = "Predict each client's label using general repayment-behavior patterns.\n\n"
OUTPUT = ('Return ONLY a JSON array, one object per client in order:\n'
          '[{"id":0,"label":0,"confidence":0.0}, ...]\n'
          "confidence = probability that label=1 (default), 0..1. No prose.")
def build(feats, sh, shy, batch, start):
    common = PRE + f"{len(sh)} labeled examples (features -> label):\n{shots_block(sh, shy, feats)}\n\n"
    clients = "\n".join(f"id {start+j}: {row_text(r, feats)}" for j, (_, r) in enumerate(batch.iterrows()))
    return common + INSTR + "Clients:\n" + clients + "\n\n" + OUTPUT

CONDS = {
    "E1_topk":     (top_feats, rand_shots, rand_y),
    "E2_boundary": (FEATURES,  bnd_shots,  bnd_y),
    "E3_combined": (top_feats, bnd_shots,  bnd_y),
    "E4_representative": (FEATURES, rep_shots, rep_y),
    "E5_topk_rep":       (top_feats, rep_shots, rep_y),
}

def run_condition(name):
    feats, sh, shy = CONDS[name]
    cache_path = f"{RESULTS}/cache_ext_{name}.json"
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    preds, confs = [], []
    for s in range(0, len(R.Xe), BATCH):
        batch = R.Xe.iloc[s:s+BATCH]
        prompt = build(feats, sh, shy, batch, s)
        key = hashlib.md5((name + str(s) + prompt).encode()).hexdigest()
        if key not in cache:
            cache[key] = R.parse(R.call_llm(prompt), len(batch), s)
            json.dump(cache, open(cache_path, "w"))
            print(f"  {name}: {s+len(batch)}/{len(R.Xe)}", flush=True)
        for lab, cf in cache[key]:
            preds.append(int(lab)); confs.append(float(cf))
    return np.array(preds), np.array(confs)

def main():
    print(f"\nTOPK={TOPK} features: {top_feats}")
    p1, pr1 = R.run_condition("L1_direct")     # baseline (cached from ablation)
    results = {"L1_direct(baseline)": (p1, pr1)}
    for nm in CONDS:
        print(f"running {nm} ...")
        results[nm] = run_condition(nm)

    rows = []
    for nm, (p, pr) in results.items():
        m = R.mvec(p, pr)
        rows.append({"condition": nm, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}})
    # include fitted RF for reference
    rfm = R.mvec(R.rf.predict(R.Xe), R.rf.predict_proba(R.Xe)[:, 1])
    rows.append({"condition": "RandomForest(ref)", **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in rfm.items()}})
    res = pd.DataFrame(rows)
    res.to_csv(f"{RESULTS}/l2b_ext_N{len(R.A)}.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(f"\n============ L2b EXTENSION (N={len(R.A)}) ============")
    print(res.to_string(index=False))

    print("\n---- paired differences vs L1 (mean [95% CI]); CI excludes 0 = significant ----")
    for nm in CONDS:
        p2, pr2 = results[nm]
        for metric in ["f1", "auc_roc", "recall"]:
            m, lo, hi = R.paired(p2, pr2, p1, pr1, metric)
            print(f"  {nm:14s} {metric:8s} {m:+.4f} [{lo:+.4f},{hi:+.4f}] {'SIG' if lo*hi>0 else 'ns'}")
    print(f"\nsaved -> {RESULTS}/l2b_ext_N{len(R.A)}.csv")

if __name__ == "__main__":
    main()
