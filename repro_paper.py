"""Faithful reproduction of research_spark_hub.py (the notebook behind the paper).
Same split, same 'Prompt 1/2/3' logic, on data/UCI_Credit_Card.csv.
Goal: confirm what Prompt 1 and Prompt 2 actually are and reproduce the numbers.
"""
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

df = pd.read_csv("data/UCI_Credit_Card.csv")
df = df.rename(columns={"default.payment.next.month": "DEFAULT"})
if "ID" in df.columns:
    df = df.drop(columns=["ID"])
df["DEFAULT"] = pd.to_numeric(df["DEFAULT"], errors="coerce")
df = df.dropna(subset=["DEFAULT"]); df["DEFAULT"] = df["DEFAULT"].astype(int)

X = df.drop(columns=["DEFAULT"]); y = df["DEFAULT"]
# NOTE: the paper notebook does NOT stratify
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

clients = X_test.iloc[:2000]
actual = y_test.iloc[:2000].tolist()

# ---- Prompt 1: row.sum() > grand mean of all cells  (labeled "direct LLM" in paper) ----
p1 = [1 if clients.iloc[i].sum() > clients.values.mean() else 0 for i in range(len(clients))]
# ---- Prompt 2: row.mean() > mean of column means    (labeled "LLM as RF" in paper) ----
p2 = [1 if clients.iloc[i].mean() > clients.mean().mean() else 0 for i in range(len(clients))]
# ---- Prompt 3: the real Random Forest ----
p3 = model.predict(clients)

def acc(pred): return 100 * sum(a == b for a, b in zip(pred, actual)) / len(actual)
def posrate(pred): return 100 * sum(pred) / len(pred)

print("class balance of the 2000 eval cases: default =",
      round(100 * sum(actual) / len(actual), 2), "%")
print()
print(f"Prompt 1 ('direct LLM' = row.sum>grandmean): acc {acc(p1):.2f}%  | predicted-default {posrate(p1):.1f}%")
print(f"Prompt 2 ('LLM as RF'  = row.mean>meanmeans): acc {acc(p2):.2f}%  | predicted-default {posrate(p2):.1f}%")
print(f"Prompt 3 (REAL Random Forest):               acc {acc(p3):.2f}%  | predicted-default {posrate(list(p3)):.1f}%")

# majority baseline
maj = [0] * len(actual)
print(f"Majority baseline (always 'no default'):     acc {acc(maj):.2f}%")

print("\n-- per-class for the two heuristics labeled as LLM --")
for name, pred in [("Prompt 1", p1), ("Prompt 2", p2)]:
    tn, fp, fn, tp = confusion_matrix(actual, pred, labels=[0, 1]).ravel()
    pr = precision_score(actual, pred, zero_division=0)
    rc = recall_score(actual, pred, zero_division=0)
    f1 = f1_score(actual, pred, zero_division=0)
    print(f"{name}: TN={tn} FP={fp} FN={fn} TP={tp} | precision={pr:.3f} recall={rc:.3f} F1={f1:.3f}")

tn, fp, fn, tp = confusion_matrix(actual, p3, labels=[0, 1]).ravel()
print(f"\nPrompt3 RF: TN={tn} FP={fp} FN={fn} TP={tp} | "
      f"precision={precision_score(actual,p3):.4f} recall={recall_score(actual,p3):.4f} F1={f1_score(actual,p3):.4f}")

lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
lp = lr.predict(clients)
print(f"LogReg: acc {acc(lp):.2f}%  recall={recall_score(actual,lp):.4f} F1={f1_score(actual,lp):.4f}")
