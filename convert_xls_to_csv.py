"""Convert the canonical UCI 'default of credit card clients' .xls into the
Kaggle-schema CSV (UCI_Credit_Card.csv) that the paper/Colab uses.

The UCI .xls has a 2-row header (X1..X23 on row 1, human labels on row 2).
The Kaggle CSV renames: ID, LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE,
PAY_0..PAY_6 (note PAY_0, not PAY_1), BILL_AMT1..6, PAY_AMT1..6,
default.payment.next.month.
"""
import pandas as pd

SRC = "data/default_of_credit_card_clients.xls"
DST = "data/UCI_Credit_Card.csv"

# row 0 is 'X1..X23 / Y' group header; row 1 is the real column names; data starts row 2
raw = pd.read_excel(SRC, header=1)

# The .xls real header names the repayment vars PAY_0, PAY_2..PAY_6 already,
# and the target 'default payment next month'. Kaggle uses dotted target name.
rename = {"default payment next month": "default.payment.next.month",
          "PAY_1": "PAY_0"}  # safety if this copy labels it PAY_1
raw = raw.rename(columns=rename)

# Kaggle canonical column order
cols = (["ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
        + [f"PAY_{i}" for i in [0, 2, 3, 4, 5, 6]]
        + [f"BILL_AMT{i}" for i in range(1, 7)]
        + [f"PAY_AMT{i}" for i in range(1, 7)]
        + ["default.payment.next.month"])

missing = [c for c in cols if c not in raw.columns]
assert not missing, f"missing columns: {missing}\nhave: {list(raw.columns)}"

df = raw[cols].copy()
df.to_csv(DST, index=False)

print("wrote", DST)
print("shape:", df.shape)
print("columns:", list(df.columns))
print("\nclass balance (default.payment.next.month):")
print(df["default.payment.next.month"].value_counts(normalize=True).round(4).to_string())
print("\ndtypes all numeric:", df.dtypes.apply(lambda t: t.kind in "if").all())
print("\nhead:")
print(df.head(3).to_string())
