"""
Repeated-runs + confidence intervals to firm up the diffusion-vs-SMOTE claim.

For each of several seeds we regenerate the v2 diffusion model, SMOTE, and the RF,
evaluate every strategy on the SAME fixed real test set, then report mean +/- 95% CI
across seeds and the PAIRED (diffusion - SMOTE) difference per metric.
"""
import numpy as np, pandas as pd, torch
from scipy import stats
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, f1_score, average_precision_score, precision_score, accuracy_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
# reuse the v2 diffusion model + fixed split
from diffusion_aug_v2 import DDPM, FEATURES, Xtr, ytr, Xte, yte, INT_COLS

SEEDS = [0, 1, 2, 3, 4]
N_SYNTH = int((ytr == 0).sum() - (ytr == 1).sum())
RESULTS = "results"

def make_synthetic(seed, n):
    torch.manual_seed(seed); np.random.seed(seed)
    real_def = Xtr[ytr == 1].values.astype(np.float32)
    qt = QuantileTransformer(output_distribution="normal",
                             n_quantiles=min(1000, real_def.shape[0]), random_state=seed).fit(real_def)
    z = torch.tensor(qt.transform(real_def), dtype=torch.float32)
    ddpm = DDPM(z.shape[1])
    ddpm.train(z)                                   # v2 defaults: 800 epochs, cosine, T=400
    syn = pd.DataFrame(qt.inverse_transform(ddpm.sample(n, z.shape[1]).numpy()), columns=FEATURES)
    for c, (lo, hi) in INT_COLS.items():
        syn[c] = syn[c].round().clip(lo, hi).astype(int)
    for c in FEATURES:
        if c.startswith(("PAY_AMT", "LIMIT_BAL")):
            syn[c] = syn[c].clip(lower=0)
    return syn

def rf_fit(Xt, yt, seed):
    return RandomForestClassifier(n_estimators=100, max_depth=12, random_state=seed,
                                  n_jobs=-1).fit(Xt, yt)

def score(model):
    pred = model.predict(Xte); proba = model.predict_proba(Xte)[:, 1]
    return dict(recall=recall_score(yte, pred), f1=f1_score(yte, pred),
                auc_pr=average_precision_score(yte, proba),
                precision=precision_score(yte, pred), accuracy=accuracy_score(yte, pred))

STRATS = ["original", "class_weight", "oversample", "SMOTE", "diffusion"]
records = {s: [] for s in STRATS}

for seed in SEEDS:
    print(f"seed {seed} ...", flush=True)
    syn = make_synthetic(seed, N_SYNTH)
    Xo, yo = RandomOverSampler(random_state=seed).fit_resample(Xtr, ytr)
    Xs, ys = SMOTE(random_state=seed).fit_resample(Xtr, ytr)
    Xd = pd.concat([Xtr, syn], ignore_index=True); yd = pd.concat([ytr, pd.Series([1]*len(syn))], ignore_index=True)

    records["original"].append(score(rf_fit(Xtr, ytr, seed)))
    records["class_weight"].append(score(RandomForestClassifier(
        n_estimators=100, max_depth=12, random_state=seed, n_jobs=-1, class_weight="balanced").fit(Xtr, ytr)))
    records["oversample"].append(score(rf_fit(Xo, yo, seed)))
    records["SMOTE"].append(score(rf_fit(Xs, ys, seed)))
    records["diffusion"].append(score(rf_fit(Xd, yd, seed)))

def ci(vals):
    a = np.array(vals); m = a.mean()
    if len(a) < 2: return m, 0.0
    h = stats.t.ppf(0.975, len(a)-1) * a.std(ddof=1) / np.sqrt(len(a))
    return m, h

METRICS = ["recall", "f1", "auc_pr", "precision", "accuracy"]
print(f"\n===== RandomForest, {len(SEEDS)} seeds, fixed real test N={len(yte)} (mean +/- 95% CI) =====")
rows = []
for s in STRATS:
    row = {"strategy": s}
    for mname in METRICS:
        m, h = ci([r[mname] for r in records[s]])
        row[mname] = f"{m:.3f} ± {h:.3f}"
    rows.append(row)
summ = pd.DataFrame(rows)
pd.set_option("display.width", 200, "display.max_columns", 20)
print(summ.to_string(index=False))
summ.to_csv(f"{RESULTS}/diffusion_cis_summary.csv", index=False)

print("\n===== PAIRED (diffusion - SMOTE) across seeds =====")
for mname in METRICS:
    diffs = np.array([records["diffusion"][i][mname] - records["SMOTE"][i][mname] for i in range(len(SEEDS))])
    m, h = ci(diffs)
    sig = "significant (CI excludes 0)" if (m - h) * (m + h) > 0 else "NOT significant"
    print(f"  {mname:10s}: {m:+.3f} ± {h:.3f}  -> {sig}   per-seed: {np.round(diffs,3).tolist()}")

# raw per-seed dump
pd.DataFrame([{**{'strategy': s, 'seed': SEEDS[i]}, **records[s][i]}
              for s in STRATS for i in range(len(SEEDS))]).to_csv(f"{RESULTS}/diffusion_cis_raw.csv", index=False)
print(f"\nsaved -> {RESULTS}/diffusion_cis_summary.csv and diffusion_cis_raw.csv")
