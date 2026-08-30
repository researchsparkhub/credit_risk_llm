"""
Improved diffusion augmentation (fair shot for the generator):
  - QuantileTransformer(output_distribution='normal') on all features
    (proper handling of heavy-tailed money columns + ordinals)
  - cosine noise schedule, longer training
Same downstream comparison as diffusion_aug.py so results are comparable.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             accuracy_score, roc_auc_score, average_precision_score,
                             confusion_matrix)
from imblearn.over_sampling import SMOTE, RandomOverSampler

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
RESULTS = "results"; os.makedirs(RESULTS, exist_ok=True)

INT_COLS = {"SEX": (1, 2), "EDUCATION": (0, 6), "MARRIAGE": (0, 3), "AGE": (18, 100),
            **{f"PAY_{i}": (-2, 8) for i in [0, 2, 3, 4, 5, 6]}}
NONNEG = [c for c in [] ]  # set below

df = pd.read_csv("data/UCI_Credit_Card.csv").rename(columns={"default.payment.next.month": "DEFAULT"}).drop(columns=["ID"])
df["DEFAULT"] = df["DEFAULT"].astype(int)
X = df.drop(columns=["DEFAULT"]); y = df["DEFAULT"]
FEATURES = list(X.columns)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

def cosine_abar(T, s=0.008):
    t = torch.linspace(0, T, T + 1)
    f = torch.cos(((t / T + s) / (1 + s)) * np.pi / 2) ** 2
    abar = f / f[0]
    return abar[1:]  # length T

class Denoiser(nn.Module):
    def __init__(self, d, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d + 1, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(),
                                 nn.Linear(h, d))
    def forward(self, x, t): return self.net(torch.cat([x, t[:, None]], 1))

class DDPM:
    def __init__(self, d, T=400):
        self.T = T
        self.abar = cosine_abar(T)
        abar_prev = torch.cat([torch.tensor([1.0]), self.abar[:-1]])
        self.betas = (1 - self.abar / abar_prev).clamp(1e-4, 0.999)
        self.alphas = 1 - self.betas
        self.model = Denoiser(d)
    def train(self, x0, epochs=800, bs=256, lr=1e-3):
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        n = x0.shape[0]; last = 0.0
        for ep in range(epochs):
            idx = torch.randperm(n)
            for s in range(0, n, bs):
                b = x0[idx[s:s+bs]]
                t = torch.randint(0, self.T, (b.shape[0],))
                ab = self.abar[t][:, None]
                noise = torch.randn_like(b)
                xt = ab.sqrt() * b + (1 - ab).sqrt() * noise
                loss = ((self.model(xt, t.float() / self.T) - noise) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                last = loss.item()
        return last
    @torch.no_grad()
    def sample(self, n, d):
        x = torch.randn(n, d)
        for t in reversed(range(self.T)):
            tt = torch.full((n,), t)
            ab, a, beta = self.abar[t], self.alphas[t], self.betas[t]
            eps = self.model(x, tt.float() / self.T)
            mean = (x - beta / (1 - ab).sqrt() * eps) / a.sqrt()
            x = mean + (beta.sqrt() * torch.randn_like(x) if t > 0 else 0)
        return x

def make_synthetic(n):
    real_def = Xtr[ytr == 1].values.astype(np.float32)
    qt = QuantileTransformer(output_distribution="normal",
                             n_quantiles=min(1000, real_def.shape[0]), random_state=SEED).fit(real_def)
    z = torch.tensor(qt.transform(real_def), dtype=torch.float32)
    ddpm = DDPM(z.shape[1]); loss = ddpm.train(z)
    syn = qt.inverse_transform(ddpm.sample(n, z.shape[1]).numpy())
    syn = pd.DataFrame(syn, columns=FEATURES)
    for c, (lo, hi) in INT_COLS.items():
        syn[c] = syn[c].round().clip(lo, hi).astype(int)
    for c in FEATURES:
        if c.startswith(("PAY_AMT", "LIMIT_BAL")):   # these are non-negative; BILL_AMT can be negative
            syn[c] = syn[c].clip(lower=0)
    return syn, loss

def synthetic_quality(syn):
    real_def = Xtr[ytr == 1]
    lines = ["Per-feature mean (real def / synth def):"]
    for c in FEATURES:
        lines.append(f"  {c:12s} {real_def[c].mean():12.2f} / {syn[c].mean():12.2f}")
    n = min(len(real_def), len(syn))
    Xd = pd.concat([real_def.sample(n, random_state=SEED), syn.sample(n, random_state=SEED)])
    yd = np.r_[np.ones(n), np.zeros(n)]
    a, b, c, d = train_test_split(Xd, yd, test_size=0.3, random_state=SEED, stratify=yd)
    det = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1).fit(a, c)
    auc = roc_auc_score(d, det.predict_proba(b)[:, 1])
    lines.append(f"\nDetection AUC (real vs synthetic; 0.5=indistinguishable): {auc:.3f}")
    return "\n".join(lines), auc

def metrics(name, model):
    pred = model.predict(Xte); proba = model.predict_proba(Xte)[:, 1]
    tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
    return dict(strategy=name, accuracy=round(100*accuracy_score(yte, pred), 2),
                precision=round(precision_score(yte, pred, zero_division=0), 4),
                recall=round(recall_score(yte, pred, zero_division=0), 4),
                f1=round(f1_score(yte, pred, zero_division=0), 4),
                auc_roc=round(roc_auc_score(yte, proba), 4),
                auc_pr=round(average_precision_score(yte, proba), 4),
                TP=int(tp), FN=int(fn), FP=int(fp))

def rf(): return RandomForestClassifier(n_estimators=100, max_depth=12, random_state=SEED, n_jobs=-1)
def lr(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))
def fit_on(Xt, yt, est): m = est(); m.fit(Xt, yt); return m

def main():
    n_pos, n_neg = int((ytr == 1).sum()), int((ytr == 0).sum())
    n_synth = n_neg - n_pos
    print(f"generating {n_synth} synthetic defaulters (improved: quantile-normal + cosine, 800 epochs)")
    syn, loss = make_synthetic(n_synth)
    print(f"  ddpm final loss {loss:.4f}")
    q_txt, det_auc = synthetic_quality(syn)
    open(f"{RESULTS}/synthetic_quality_v2.txt", "w").write(q_txt)
    print(f"  detection AUC = {det_auc:.3f}  (v1 was 0.966; lower is better)")

    Xt_over, yt_over = RandomOverSampler(random_state=SEED).fit_resample(Xtr, ytr)
    Xt_smote, yt_smote = SMOTE(random_state=SEED).fit_resample(Xtr, ytr)
    Xt_diff = pd.concat([Xtr, syn], ignore_index=True)
    yt_diff = pd.concat([ytr, pd.Series([1]*len(syn))], ignore_index=True)

    rows = []
    for name, est in [("RandomForest", rf), ("LogReg", lr)]:
        rows.append(metrics(f"{name}: original", fit_on(Xtr, ytr, est)))
        if name == "RandomForest":
            m = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=SEED, n_jobs=-1, class_weight="balanced").fit(Xtr, ytr)
        else:
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED, class_weight="balanced")).fit(Xtr, ytr)
        rows.append(metrics(f"{name}: class_weight", m))
        rows.append(metrics(f"{name}: oversample", fit_on(Xt_over, yt_over, est)))
        rows.append(metrics(f"{name}: SMOTE", fit_on(Xt_smote, yt_smote, est)))
        rows.append(metrics(f"{name}: diffusion", fit_on(Xt_diff, yt_diff, est)))
    res = pd.DataFrame(rows)
    res.to_csv(f"{RESULTS}/diffusion_aug_results_v2.csv", index=False)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(f"\n=========== IMPROVED DIFFUSION AUG (real test N={len(yte)}, default {100*yte.mean():.1f}%) ===========")
    print(res.to_string(index=False))
    print(f"\nsaved -> {RESULTS}/diffusion_aug_results_v2.csv")

if __name__ == "__main__":
    main()
