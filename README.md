# Week 2 Mini‑Assignment - First Data Analysis (Commodities) - Tony

A step‑by‑step tutorial README for the notebook **`702_A2.ipynb`**. 

It walks you through what the project does, the dataset it uses, how to set up your environment, and how to run and extend the analysis.


## 1) Overview

This notebook takes a daily historical **commodities futures** dataset (e.g., Gold, Silver, Platinum, Palladium, etc.) and shows how to ingest, process, visualize, and run basic ML.

---

## 2) Dataset description

The notebook expects a single CSV file (found in the repository) named **`all_commodities_data.csv`**, with the following columns.

- **`Date`** *(YYYY‑MM‑DD)* — trading day.
- **`Open`** — opening price.
- **`High`** — high price of the day.
- **`Low`** — low price of the day.
- **`Close`** — closing price.
- **`Volume`** — trading volume.
- **`Commodity`** — commodity name (e.g., `"Gold"`, `"Silver"`).
- **`Ticker`** — commodity code (e.g., `"GC"`).

### Example data dictionary

| Column     | Type      | Description                                              |
|------------|-----------|----------------------------------------------------------|
| Date       | date      | Trading date (YYYY‑MM‑DD).                               |
| Open       | float     | Opening price.                                           |
| High       | float     | Intraday high.                                           |
| Low        | float     | Intraday low.                                            |
| Close      | float     | Closing price.                                           |
| Volume     | int/float | Trading volume (may be missing for some contracts).      |
| Commodity  | string    | Commodity group or name (e.g., Gold, Silver, etc.).      |
| Ticker     | string    | Contract ticker/symbol (optional).                       |

---

## 3) Environment setup

You can use **conda** or **pip**. Python 3.10+ is recommended.

### Option A — Conda
```bash
# create & activate
conda create -n ids706-assign2 python=3.11 -y
conda activate ids706-assign2

# install dependencies
conda install -y pandas numpy matplotlib seaborn scikit-learn
```

### Option B — Pip (virtualenv)
```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scikit-learn
```

> Launch Jupyter (or VS Code, or your preferred IDE) in this environment.

---

## 4) Importing the dataset

1. Place `all_commodities_data.csv` into the `data/` folder.
2. In the notebook, change the CSV read line to a **relative path**:

```python
import pandas as pd

# recommended: relative path under ./data
df = pd.read_csv("data/all_commodities_data.csv")
```

3. **Parse the date column** and **sort**:
```python
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Commodity", "Date"]).reset_index(drop=True)
```

4. **Quick sanity checks**:
```python
display(df.head())
display(df.info())
display(df.describe())

print("Commodities:", df["Commodity"].unique())
print("Date range:", df["Date"].min(), "→", df["Date"].max())
```

---

## 5) Exploratory data analysis (EDA)

You’ll see in the notebook some rudimentary EDA. Feel free to dive deeper into the data.

- Summary statistics with `df.describe()`.
- Count of rows per commodity / ticker:
  ```python
  df.groupby("Commodity")["Close"].size().sort_values(ascending=False)
  ```
- Missing values check:
  ```python
  df.isna().mean().sort_values(ascending=False)
  ```

---

## 6) Visualizations

### A) Clustered High/Low time‑series by commodity
The notebook creates a grid of subplots—**one small chart per commodity**—plotting **High** and **Low** over time, plus a panel for **overall averages** across commodities. This gives you a quick, side‑by‑side sense of level and volatility.

A 6th panel for average High/Low for all metals has been included.

---

## 7) ML

### A) Classification — “Will price go up or down tomorrow?”

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

use_cols = ["ret_1d", "price_vs_ma7", "close_lag1", "close_lag5"]
X = df[use_cols].dropna()
y = df.loc[X.index, "up_next"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # preserve time ordering for simplicity
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_s, y_train)

y_pred = clf.predict(X_test_s)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

### B) Regression — “What is tomorrow’s price point?”

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

reg_use_cols = ["ret_1d", "price_vs_ma7", "close_lag1", "close_lag5"]
Xr = df[reg_use_cols].dropna()
yr = df.loc[Xr.index, "close_tomorrow"]

# simple chronological split
split = int(len(Xr) * 0.8)
Xr_train, Xr_test = Xr.iloc[:split], Xr.iloc[split:]
yr_train, yr_test = yr.iloc[:split], yr.iloc[split:]

reg = GradientBoostingRegressor(random_state=42)
reg.fit(Xr_train, yr_train)

yr_pred = reg.predict(Xr_test)
rmse = mean_squared_error(yr_test, yr_pred, squared=False)
r2   = r2_score(yr_test, yr_pred)
print("RMSE:", rmse, "| R²:", r2)
```

---

## 8) Running the notebook

1. Open the .ipynb in Jupyter.
2. Edit the data path to `data/all_commodities_data.csv`.
3. Run cells **top‑to‑bottom**. If a plot grid overflows your number of commodities, hide unused axes as shown in the code.
4. Review the outputs (tables, charts, metrics).

---
