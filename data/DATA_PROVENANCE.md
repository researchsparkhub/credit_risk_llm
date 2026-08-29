# Dataset provenance

**Dataset:** Default of Credit Card Clients (Taiwan, 2005)
**Canonical source (UCI):** https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls
**Downloaded:** 2026-08-29
**Original .xls md5:** 25a64500f543274fd6c9db905bfeb968

`UCI_Credit_Card.csv` was produced from the official UCI `.xls` by
`convert_xls_to_csv.py`, renamed to the standard Kaggle schema
(`uciml/default-of-credit-card-clients-dataset`) so it matches the paper/Colab
(`ID, LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0, PAY_2..PAY_6,
BILL_AMT1..6, PAY_AMT1..6, default.payment.next.month`).

- Shape: 30,000 rows x 25 columns (ID + 23 features + target)
- Class balance: 0 (no default) 77.88% / 1 (default) 22.12%
- All columns numeric

The .xls is gitignored (5.3 MB, binary); regenerate the CSV anytime with
`python convert_xls_to_csv.py`.
