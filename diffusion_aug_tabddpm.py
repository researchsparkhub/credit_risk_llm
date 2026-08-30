"""
Proper mixed-type TabDDPM for synthetic defaulters:
  - continuous cols: Gaussian diffusion in quantile-normal space
  - categorical cols (SEX, EDUCATION, MARRIAGE, PAY_0/2..6): MULTINOMIAL diffusion
    (Hoogeboom et al.; the TabDDPM recipe) -- the repayment codes are treated as
    true categoricals rather than continuous.
Same downstream comparison as before so results are directly comparable.
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.model_selection import train_test_split
# reuse the exact split + eval harness from v2 (module-level data load; main is guarded)
from diffusion_aug_v2 import (Xtr, ytr, Xte, yte, FEATURES, metrics, rf, lr, fit_on,
                              SEED, RESULTS)

torch.manual_seed(SEED); np.random.seed(SEED)

CONT = ["LIMIT_BAL", "AGE"] + [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)]
CAT  = ["SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

def cosine_abar(T, s=0.008):
    t = torch.linspace(0, T, T + 1)
    f = torch.cos(((t / T + s) / (1 + s)) * np.pi / 2) ** 2
    return (f / f[0])[1:]

class MixedDenoiser(nn.Module):
    def __init__(self, n_cont, cards, h=512):
        super().__init__()
        self.n_cont, self.cards = n_cont, cards
        d = n_cont + sum(cards)
        self.body = nn.Sequential(nn.Linear(d + 1, h), nn.SiLU(),
                                  nn.Linear(h, h), nn.SiLU(),
                                  nn.Linear(h, h), nn.SiLU())
        self.head_cont = nn.Linear(h, n_cont)
        self.heads_cat = nn.ModuleList([nn.Linear(h, k) for k in cards])
    def forward(self, x, t):
        z = self.body(torch.cat([x, t[:, None]], 1))
        return self.head_cont(z), [head(z) for head in self.heads_cat]

class TabDDPM:
    def __init__(self, n_cont, cards, T=400):
        self.T, self.n_cont, self.cards = T, n_cont, cards
        self.abar = cosine_abar(T)
        self.abar_prev = torch.cat([torch.tensor([1.0]), self.abar[:-1]])
        self.alphas = self.abar / self.abar_prev
        self.betas = (1 - self.alphas).clamp(1e-4, 0.999)
        self.model = MixedDenoiser(n_cont, cards)

    def _onehot(self, idx):  # list of (N,) long -> concatenated one-hot (N, sum cards)
        return torch.cat([F.one_hot(idx[c], k).float() for c, k in enumerate(self.cards)], 1)

    def _cat_forward(self, x0_idx, t):  # multinomial q(x_t|x_0): keep w.p. abar else uniform
        out = []
        for c, k in enumerate(self.cards):
            ab = self.abar[t]
            keep = (torch.rand(len(t)) < ab)
            rnd = torch.randint(0, k, (len(t),))
            out.append(torch.where(keep, x0_idx[c], rnd))
        return out

    def train(self, xc, xcat_idx, epochs=1000, bs=512, lr=1e-3):
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        n = xc.shape[0]; last = 0.0
        for ep in range(epochs):
            perm = torch.randperm(n)
            for s in range(0, n, bs):
                b = perm[s:s+bs]
                t = torch.randint(0, self.T, (len(b),))
                ab = self.abar[t][:, None]
                # continuous
                cc = xc[b]; eps = torch.randn_like(cc)
                xt_c = ab.sqrt() * cc + (1 - ab).sqrt() * eps
                # categorical forward
                x0i = [xcat_idx[c][b] for c in range(len(self.cards))]
                xti = self._cat_forward(x0i, t)
                xin = torch.cat([xt_c, self._onehot(xti)], 1)
                pred_c, logits = self.model(xin, t.float() / self.T)
                loss = ((pred_c - eps) ** 2).mean()
                for c in range(len(self.cards)):
                    loss = loss + F.cross_entropy(logits[c], x0i[c])
                opt.zero_grad(); loss.backward(); opt.step(); last = loss.item()
        return last

    @torch.no_grad()
    def sample(self, n):
        xc = torch.randn(n, self.n_cont)
        xi = [torch.randint(0, k, (n,)) for k in self.cards]
        for t in reversed(range(self.T)):
            tt = torch.full((n,), t)
            xin = torch.cat([xc, self._onehot(xi)], 1)
            pred_c, logits = self.model(xin, tt.float() / self.T)
            # continuous reverse
            ab, a, beta = self.abar[t], self.alphas[t], self.betas[t]
            mean = (xc - beta / (1 - ab).sqrt() * pred_c) / a.sqrt()
            xc = mean + (beta.sqrt() * torch.randn_like(xc) if t > 0 else 0)
            # categorical reverse (multinomial posterior)
            new = []
            for c, k in enumerate(self.cards):
                phat = F.softmax(logits[c], 1)
                if t > 0:
                    xt_oh = F.one_hot(xi[c], k).float()
                    a_t = self.alphas[t]; abp = self.abar_prev[t]
                    term_xt = a_t * xt_oh + (1 - a_t) / k
                    term_x0 = abp * phat + (1 - abp) / k
                    post = term_xt * term_x0
                    post = post / post.sum(1, keepdim=True)
                    new.append(torch.multinomial(post, 1).squeeze(1))
                else:
                    new.append(phat.argmax(1))
            xi = new
        return xc.numpy(), [v.numpy() for v in xi]

def make_synthetic(n):
    real_def = Xtr[ytr == 1]
    # continuous -> quantile-normal
    qt = QuantileTransformer(output_distribution="normal",
                             n_quantiles=min(1000, len(real_def)), random_state=SEED).fit(real_def[CONT])
    xc = torch.tensor(qt.transform(real_def[CONT]), dtype=torch.float32)
    # categorical -> index maps
    cats = {c: sorted(real_def[c].unique()) for c in CAT}
    idxmap = {c: {v: i for i, v in enumerate(cats[c])} for c in CAT}
    cards = [len(cats[c]) for c in CAT]
    xcat_idx = [torch.tensor(real_def[c].map(idxmap[c]).values, dtype=torch.long) for c in CAT]

    ddpm = TabDDPM(len(CONT), cards)
    loss = ddpm.train(xc, xcat_idx)
    sc, si = ddpm.sample(n)
    cont = qt.inverse_transform(sc)
    out = pd.DataFrame(index=range(n))
    for j, c in enumerate(CONT):
        out[c] = cont[:, j]
    for c, col in enumerate(CAT):
        inv = {i: v for v, i in idxmap[col].items()}
        out[col] = pd.Series(si[c]).map(inv).values
    out = out[FEATURES]
    out["AGE"] = out["AGE"].round().clip(18, 100).astype(int)
    for c in ["LIMIT_BAL"] + [f"PAY_AMT{i}" for i in range(1, 7)]:
        out[c] = out[c].clip(lower=0)
    for c in CAT:
        out[c] = out[c].astype(int)
    return out, loss

def detection_auc(syn):
    real_def = Xtr[ytr == 1]
    n = min(len(real_def), len(syn))
    Xd = pd.concat([real_def.sample(n, random_state=SEED), syn.sample(n, random_state=SEED)])
    yd = np.r_[np.ones(n), np.zeros(n)]
    a, b, c, d = train_test_split(Xd, yd, test_size=0.3, random_state=SEED, stratify=yd)
    det = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1).fit(a, c)
    return roc_auc_score(d, det.predict_proba(b)[:, 1])

def main():
    n_synth = int((ytr == 0).sum() - (ytr == 1).sum())
    print(f"generating {n_synth} synthetic defaulters (mixed-type TabDDPM: Gaussian + multinomial)")
    syn, loss = make_synthetic(n_synth)
    print(f"  final loss {loss:.4f}")
    auc = detection_auc(syn)
    print(f"  detection AUC = {auc:.3f}   (v1=0.966, v2=0.799; lower is better)")
    lines = ["Per-feature mean (real def / synth def):"]
    for c in FEATURES:
        lines.append(f"  {c:12s} {Xtr[ytr==1][c].mean():12.2f} / {syn[c].mean():12.2f}")
    lines.append(f"\nDetection AUC: {auc:.3f}")
    open(f"{RESULTS}/synthetic_quality_tabddpm.txt", "w").write("\n".join(lines))

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
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED, class_weight="balanced")).fit(Xtr, ytr)
        rows.append(metrics(f"{name}: class_weight", m))
        rows.append(metrics(f"{name}: oversample", fit_on(Xt_over, yt_over, est)))
        rows.append(metrics(f"{name}: SMOTE", fit_on(Xt_smote, yt_smote, est)))
        rows.append(metrics(f"{name}: diffusion(TabDDPM)", fit_on(Xt_diff, yt_diff, est)))
    res = pd.DataFrame(rows)
    res.to_csv(f"{RESULTS}/diffusion_aug_results_tabddpm.csv", index=False)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(f"\n=========== MIXED-TYPE TabDDPM (real test N={len(yte)}, default {100*yte.mean():.1f}%) ===========")
    print(res.to_string(index=False))
    print(f"\nsaved -> {RESULTS}/diffusion_aug_results_tabddpm.csv")

if __name__ == "__main__":
    main()
