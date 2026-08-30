"""
Inverse hybrid: feed the LLM's E1 output (label + confidence) into RF as extra
features (stacking) and test whether RF's recall / AUC improves.

We have LLM (E1) predictions only for the 2000 eval cases, so we evaluate by
5-fold stratified CV ON those 2000: within each fold compare
   RF_base  = RF(23 raw features)
   RF_stack = RF(23 raw features + LLM_label + LLM_conf)
trained on the same folds -> out-of-fold metrics. Only difference = LLM feature.
(Absolute RF numbers are lower than the 24k-trained RF because each fold trains on
~1600 rows; the controlled comparison is RF_stack vs RF_base.)
"""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import run_ablation as R
import run_l2b_ext as X

Xe = R.Xe.reset_index(drop=True); A = R.A
e1_pred, e1_conf = X.run_condition("E1_topk")          # cached; aligned to Xe order
e1_pred = np.asarray(e1_pred); e1_conf = np.asarray(e1_conf)

base_cols = list(Xe.columns)
Xaug = Xe.copy()
Xaug["LLM_label"] = e1_pred
Xaug["LLM_conf"] = e1_conf

def rf(): return RandomForestClassifier(n_estimators=100, max_depth=12, random_state=R.SEED, n_jobs=-1)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=R.SEED)
oof = {k: np.zeros(len(A)) for k in ["base_pred", "base_proba", "stack_pred", "stack_proba"]}
for tr, te in skf.split(Xe, A):
    b = rf().fit(Xe.iloc[tr], A[tr])
    oof["base_pred"][te] = b.predict(Xe.iloc[te]); oof["base_proba"][te] = b.predict_proba(Xe.iloc[te])[:, 1]
    s = rf().fit(Xaug.iloc[tr], A[tr])
    oof["stack_pred"][te] = s.predict(Xaug.iloc[te]); oof["stack_proba"][te] = s.predict_proba(Xaug.iloc[te])[:, 1]

rows = []
conds = {
    "LLM E1 alone":            (e1_pred, e1_conf),
    "RF_base (23 feats, CV)":  (oof["base_pred"].astype(int), oof["base_proba"]),
    "RF_stack (+LLM, CV)":     (oof["stack_pred"].astype(int), oof["stack_proba"]),
    "RF (24k-trained, ref)":   (R.rf.predict(R.Xe), R.rf.predict_proba(R.Xe)[:, 1]),
}
for nm, (p, pr) in conds.items():
    m = R.mvec(np.asarray(p), np.asarray(pr))
    rows.append({"condition": nm, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}})
res = pd.DataFrame(rows)
res.to_csv(f"{R.RESULTS}/inverse_stack_N{len(A)}.csv", index=False)
pd.set_option("display.width", 220, "display.max_columns", 20)
print(f"============ INVERSE HYBRID: LLM-as-feature-for-RF (5-fold CV on N={len(A)}) ============")
print(res.to_string(index=False))
print("\nRF feature importance of the LLM inputs (last fold's stack model):")
imp = pd.Series(s.feature_importances_, index=list(Xaug.columns)).sort_values(ascending=False)
print(imp.head(6).round(4).to_string())

print("\n---- paired RF_stack - RF_base (mean [95% CI]); CI excludes 0 = significant ----")
for metric in ["recall", "f1", "auc_roc"]:
    m, lo, hi = R.paired(oof["stack_pred"].astype(int), oof["stack_proba"],
                         oof["base_pred"].astype(int), oof["base_proba"], metric)
    print(f"  {metric:8s} {m:+.4f} [{lo:+.4f},{hi:+.4f}] {'SIG' if lo*hi>0 else 'ns'}")
print(f"\nsaved -> {R.RESULTS}/inverse_stack_N{len(A)}.csv")
