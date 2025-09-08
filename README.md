# Week 2 Mini‑Assignment — First Data Analysis (Commodities)

A step‑by‑step tutorial README for the notebook **`702_A2.ipynb`**. It walks you through what the project does, the dataset it uses, how to set up your environment, and how to run and extend the analysis.

---

## 1) What this project does

This notebook takes a daily historical **commodities futures** dataset (e.g., Gold, Silver, Platinum, Palladium, etc.) and shows a complete mini‑pipeline:

1. **Load & inspect** a CSV dataset of multi‑commodity prices.
2. **Clean & prepare** the data (parse dates, sort by time, filter per commodity).
3. **Explore** the data with quick summary statistics and visualizations.
4. **Visualize** price ranges over time (per commodity) and the **average High/Low** across all commodities.
5. **Engineer features** such as returns, moving averages, and simple lags.
6. **Model** two simple predictive tasks (meant to be illustrative, not production‑grade):
   - **Classification:** predict whether price will go **up or down** based on recent movement (e.g., using `LogisticRegression`).
   - **Regression:** predict a numeric **next‑day price** (e.g., using `GradientBoostingRegressor`).
7. **Evaluate** with intuitive metrics (accuracy, confusion matrix for classification; RMSE, R² for regression).
8. **Discuss** limitations and how to iterate further.

> The focus is on getting end‑to‑end practice with data handling, EDA, feature engineering, and simple ML baselines on time‑ordered financial data.

---

## 2) Dataset description

The notebook expects a single CSV file, commonly named **`all_commodities_data.csv`**, with at least the following columns (case may vary; the code demonstrates both lowercase and Title‑Case usage):

- **`Date`** *(YYYY‑MM‑DD)* — trading day.
- **`Open`** — opening price.
- **`High`** — high price of the day.
- **`Low`** — low price of the day.
- **`Close`** — closing price.
- **`Volume`** — trading volume (if available).
- **`Commodity`** — human‑readable name (e.g., `"Gold"`, `"Silver"`).
- **`Ticker`** *(optional)* — an instrument code (e.g., `"GC"`).

> If your file uses lowercase (e.g., `date, open, high, low, close, volume, commodity, ticker`) that’s fine—just make sure the column names used in the notebook match your file.

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

## 3) Project structure (suggested)

```
project-root/
├─ data/
│  └─ all_commodities_data.csv         # your dataset (put it here)
├─ 702_A2.ipynb                         # the notebook
└─ README.md                            # this file
```

> The original notebook referenced a user‑specific, absolute path (e.g., `/Users/.../all_commodities_data.csv`). For portability, use a **relative path** like `data/all_commodities_data.csv` instead (instructions below).

---

## 4) Environment setup

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

## 5) Importing the dataset

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

> If your file uses lowercase column names, either rename them or adjust the code accordingly (e.g., `df.rename(columns=str.title, inplace=True)`).

---

## 6) Exploratory data analysis (EDA)

Examples you’ll see in the notebook:

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

## 7) Visualizations

### A) Clustered High/Low time‑series by commodity
The notebook creates a grid of subplots—**one small chart per commodity**—plotting **High** and **Low** over time, plus a panel for **overall averages** across commodities. This gives you a quick, side‑by‑side sense of level and volatility.

> Tip: If you add a 6th panel for average High/Low, compute it from a pivot table (by date) and plot in the last axis.

### B) Other quick visuals (optional)
- Distribution of daily returns (histogram).
- Boxplots of daily ranges `(High‑Low)` by commodity.
- Correlations between engineered features.

---

## 8) Feature engineering (illustrative)

A lightweight, time‑aware set of features you can build per commodity:

```python
# sort within each commodity
df = df.sort_values(["Commodity", "Date"]).reset_index(drop=True)

# daily % return from close
df["ret_1d"] = df.groupby("Commodity")["Close"].pct_change(1)

# simple lags on close
df["close_lag1"] = df.groupby("Commodity")["Close"].shift(1)
df["close_lag5"] = df.groupby("Commodity")["Close"].shift(5)

# moving average & position vs MA
df["ma_7"] = df.groupby("Commodity")["Close"].transform(lambda s: s.rolling(7).mean())
df["price_vs_ma7"] = (df["Close"] - df["ma_7"]) / df["ma_7"]

# classification target: up(1)/down(0) relative to *tomorrow*
df["close_tomorrow"] = df.groupby("Commodity")["Close"].shift(-1)
df["up_next"] = (df["close_tomorrow"] > df["Close"]).astype("int")
```

> **Important:** We keep all operations **within commodity groups** to prevent look‑ahead leakage and cross‑series mixing.

---

## 9) Baseline models (illustrative)

### A) Classification — “Will price go up tomorrow?”

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

### B) Regression — “What is tomorrow’s close?”

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

> These are **didactic baselines**. For serious forecasting, consider walk‑forward validation, richer features, and domain‑appropriate models.

---

## 10) Running the notebook

1. Open `702_A2.ipynb` in Jupyter.
2. Edit the data path to `data/all_commodities_data.csv`.
3. Run cells **top‑to‑bottom**. If a plot grid overflows your number of commodities, hide unused axes as shown in the code.
4. Review the outputs (tables, charts, metrics).

---
