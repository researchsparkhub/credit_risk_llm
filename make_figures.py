"""Generate paper figures: (1) recall-vs-AUC trade-off scatter, (2) ROC & PR curves."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
import run_ablation as R
import run_l2b_ext as X
import run_hybrid as H

OUT = "paper"
A = R.A

# proba/pred vectors for curves
p_l1, c_l1 = R.run_condition("L1_direct")
p_e1, c_e1 = X.run_condition("E1_topk")
p_e6, c_e6 = H.run()
rf_pr = R.rf.predict_proba(R.Xe)[:, 1]
curves = {
    "L1 direct LLM": np.asarray(c_l1),
    "E1 top-8 feats": np.asarray(c_e1),
    "E6 hybrid (RF$\\rightarrow$LLM)": np.asarray(c_e6),
    "Random Forest": rf_pr,
}
colors = {"L1 direct LLM": "#d62728", "E1 top-8 feats": "#ff7f0e",
          "E6 hybrid (RF$\\rightarrow$LLM)": "#2ca02c", "Random Forest": "#1f77b4"}

# ---------- Figure 2: ROC + PR ----------
fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.1))
for name, proba in curves.items():
    fpr, tpr, _ = roc_curve(A, proba)
    ax[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(A, proba):.2f})", color=colors[name], lw=1.8)
    pr, rc, _ = precision_recall_curve(A, proba)
    ax[1].plot(rc, pr, label=f"{name} ({average_precision_score(A, proba):.2f})", color=colors[name], lw=1.8)
ax[0].plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate"); ax[0].set_title("ROC (AUC in legend)")
ax[0].legend(fontsize=7, loc="lower right")
ax[1].axhline(A.mean(), ls="--", color="k", lw=0.8, alpha=0.5)
ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_title("Precision-Recall (AP in legend)")
ax[1].legend(fontsize=7, loc="upper right")
for a in ax: a.grid(alpha=0.25); a.set_xlim(0, 1)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_roc_pr.pdf"); plt.close(fig)
print("wrote fig_roc_pr.pdf")

# ---------- Figure 1: recall vs AUC-ROC trade-off ----------
def read(csv, cond):
    df = pd.read_csv(csv)
    r = df[df["condition"] == cond].iloc[0]
    return float(r["recall"]), float(r["auc_roc"])

pts = {}  # label -> (recall, auc, group)
abl = "results/ablation_N2000.csv"
ext = "results/l2b_ext_N2000.csv"
hyb = "results/hybrid_N2000.csv"
ens = "results/ensemble_hybrid_N2000.csv"
pts["LogReg"] = (*read(abl, "LogReg"), "fitted")
pts["Random Forest"] = (*read(abl, "RandomForest"), "fitted")
pts["AdaBoost"] = (*read(ens, "AdaBoost"), "fitted")
pts["L1 direct"] = (*read(abl, "L1_direct"), "llm")
pts["L2 imitate"] = (*read(abl, "L2_full"), "llm")
pts["E1 top-8"] = (*read(ext, "E1_topk"), "llm")
pts["E2 boundary"] = (*read(ext, "E2_boundary"), "llm")
pts["E6 hybrid"] = (*read(hyb, "E6_hybrid(E1+RF)"), "hybrid")
pts["E7 ensemble"] = (*read(ens, "E7_hybrid (E1+RF+LR+ADA)"), "hybrid")

gcolor = {"fitted": "#1f77b4", "llm": "#d62728", "hybrid": "#2ca02c"}
gmark = {"fitted": "s", "llm": "o", "hybrid": "*"}
fig, ax = plt.subplots(figsize=(3.4, 3.0))
seen = set()
for label, (rec, auc, g) in pts.items():
    ax.scatter(auc, rec, c=gcolor[g], marker=gmark[g], s=90 if g == "hybrid" else 45,
               edgecolor="k", linewidth=0.4, zorder=3, label=g if g not in seen else None)
    seen.add(g)
    dy = 0.006 if label != "E6 hybrid" else 0.010
    ax.annotate(label, (auc, rec), fontsize=6, xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("AUC-ROC (ranking) $\\rightarrow$ better")
ax.set_ylabel("Recall (defaulters caught) $\\rightarrow$ better")
ax.set_title("Recall vs.\\ ranking trade-off", fontsize=9)
ax.annotate("best", (0.79, 0.55), fontsize=8, color="gray")
ax.grid(alpha=0.25)
handles = [plt.Line2D([0], [0], marker=gmark[k], color="w", markerfacecolor=gcolor[k],
                      markeredgecolor="k", markersize=8, label=k) for k in gmark]
ax.legend(handles=handles, fontsize=7, loc="lower center", ncol=3, columnspacing=0.8, handletextpad=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_tradeoff.pdf"); plt.close(fig)
print("wrote fig_tradeoff.pdf")
