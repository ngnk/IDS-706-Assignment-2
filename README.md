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
| Ticker     | string    | Contract ticker/symbol                                   |

---

## 3) Environment setup

You can use **conda** or **pip**. Python 3.10+ is recommended.

# create & activate
conda create -n ids706-assign2 python=3.11 -y
conda activate ids706-assign2

# install dependencies
conda install -y pandas numpy matplotlib seaborn scikit-learn
# OR
pip install pandas numpy matplotlib seaborn scikit-learn

> Launch Jupyter (or VS Code, or your preferred IDE) in this environment.

---

## 4) Importing the dataset

1. Note the path for the file `all_commodities_data.csv`
2. In the notebook, change the CSV read line to a **relative path**:

```python
import pandas as pd
df = pd.read_csv("your_path_here")
```

---

## 5) Inspections & Exploratory data analysis (EDA)

You’ll see in the notebook some rudimentary inspection and EDA.

- Display dataframe
```python
df.head(10) # display the first 10 rows
```

- dataframe info
```python
df.info() # information on column datatypes, there are no nulls
```

- Description
```python
df.describe() # descriptive statistics
```

- Shape
```python
df.shape
```

And so on. More examples can be seen within the notebook.

For basic grouping and filtering, see examples of the following implementations.
- Group by year
```python
# group by year
df['date'] = pd.to_datetime(df['date']) # convert to datetime
df['year'] = df['date'].dt.year

# Group by year and aggregate (example: mean close price per year)
grouped = df.groupby('year').agg({
    'open': 'mean',
    'high': 'mean',
    'low': 'mean',
    'close': 'mean',
    'volume': 'sum'
}).reset_index()

print(grouped)
```

- Group by metal
```python
# Group by commodity
grouped = df.groupby('commodity').agg({
    'open': 'mean',
    'high': 'mean',
    'low': 'mean',
    'close': 'mean',
    'volume': 'sum'
}).reset_index()

print(grouped)
```

- Group by Gold by year
```python
# Group by Gold per year
gold_df = df[df['commodity'] == 'Gold'].copy()

gold_df['date'] = pd.to_datetime(gold_df['date'])
gold_df['year'] = gold_df['date'].dt.year

gold_grouped = gold_df.groupby('year').agg({
    'open': 'mean',
    'high': 'mean',
    'low': 'mean',
    'close': 'mean',
    'volume': 'sum'
}).reset_index()

print(gold_grouped)
```

Play around with the groupings so mix and match the data as you'd like.
These groups come in key for the next section (ML)

---
## 6) ML

### A)  Regression — “What is tomorrow’s price point?”

The following pipeline attempts to predict the next day's pricepoint for gold, based on a the previous 5 days. 

It uses a Gradient Boosting (GB) regressor.

The model performs poorly, with a high RMSE.

```python
n_lags = 5  # # Create lag features for OPEN only
for lag in range(1, n_lags + 1):
    gold_df[f'open_lag_{lag}'] = gold_df['open'].shift(lag)

keep_cols = ['date', 'open'] + [c for c in gold_df.columns if c.startswith('open_lag_')]
gold_df = gold_df[keep_cols].dropna()

# Train/test
X = gold_df.drop(columns=['date', 'open'])
y = gold_df['open']
split_idx = int(len(X) * 0.8)  # 80/20 split
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
dates_test = gold_df['date'].iloc[split_idx:]  # for plotting

# Train
gbr = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.9,
    random_state=42
)
gbr.fit(X_train, y_train)

# Evaluate
y_pred = gbr.predict(X_test)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
r2 = float(r2_score(y_test, y_pred))

results_table = pd.DataFrame({
    "Metric": ["RMSE", "R²"],
    "Value": [rmse, r2]
})
print("\n=== Model Performance ===")
print(results_table.to_string(index=False))

# Visualize
plt.figure(figsize=(12,6))
plt.plot(dates_test, y_test.values, label="Actual Open", linewidth=2)
plt.plot(dates_test, y_pred, "--", label="Predicted Open")
plt.title("Gold Opening Price — Actual vs Predicted")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()
```

### B) Classification — “Will price go up or down tomorrow?”

This model performs significantly between, and instead of predicting the upcoming price, it simply predicts whether price will go up or down, based on the changes in previous days.

Some feature engineering was done to create the predictive features.

It predicts price increases extremely well, but all incorrectly flags decreases as increases incredibly often i.e. it is biased to price increases.

```python
gold_df = df[df['commodity'] == 'Gold'].copy() # Group by Gold per year
gold_df['date'] = pd.to_datetime(gold_df['date'])
gold_df['year'] = gold_df['date'].dt.year

gold_df = gold_df.drop(columns=['ticker','commodity','high','low','volume','year', 'open'])

gold_df['change'] = gold_df['close'].diff().apply(lambda x: 'up' if x > 0 else 'down') # Create the 'change' column based on price movement from previous day
gold_df = gold_df.drop(gold_df.index[0]) # Drop the first row (since it doesn't have a previous day to compare to)
gold_df = gold_df.reset_index(drop=True)

# Create features for the model
def create_features(df):
    df = df.copy()
    
    # Only 2 key predictors
    df['ma_7'] = df['close'].rolling(window=7).mean()
    df['price_vs_ma7'] = (df['close'] - df['ma_7']) / df['ma_7']  # Position vs 7-day MA
    df['price_change_5d'] = df['close'].pct_change(5)  # 5-day percentage change
    
    return df

# Prepare the data
gold_df_features = create_features(gold_df)
gold_df_features['target'] = gold_df_features['change'].shift(-1) # Create target variable (next day's direction)
gold_df_features = gold_df_features.dropna() # Drop rows with missing values
feature_columns = ['price_vs_ma7', 'price_change_5d'] # Select only 2 features

X = gold_df_features[feature_columns]
y = gold_df_features['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) # Split the data
scaler = StandardScaler() # Scale the features
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=42) # Train the model
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled) # Make predictions

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=['down', 'up'])

print(f"\nModel Accuracy: {accuracy:.3f}")

# Extract TP, TN, FP, FN from confusion matrix
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion Matrix Breakdown:")
print(f"True Negatives (TN - correctly predicted 'down'): {tn}")
print(f"False Positives (FP - predicted 'up', actually 'down'): {fp}")
print(f"False Negatives (FN - predicted 'down', actually 'up'): {fn}")
print(f"True Positives (TP - correctly predicted 'up'): {tp}")
```

---

## 7) Visualizations

### A) Clustered High/Low time‑series by commodity
The notebook creates a grid of subplots—**one small chart per commodity**—plotting **High** and **Low** over time, plus a panel for **overall averages** across commodities. This gives you a quick, side‑by‑side sense of level and volatility.

A 6th panel for average High/Low for all metals has been included.

```python
df['date'] = pd.to_datetime(df['date'])
commodities = df['commodity'].unique()

# Remove sharex=True to allow individual x-axis labels
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes_flat = axes.flatten()

# Plot first 5 commodities
for i, commodity in enumerate(commodities):
    if i < 5:  # Plot first 5 commodities in first 5 subplots
        ax = axes_flat[i]
        sub_df = df[df['commodity'] == commodity]
        ax.plot(sub_df['date'], sub_df['high'], label="High", color="red")
        ax.plot(sub_df['date'], sub_df['low'], label="Low", color="blue")
        ax.set_title(f"{commodity}")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        
        # Format x-axis to show years
        ax.xaxis.set_major_locator(mdates.YearLocator(base=4))  # Every 4 years
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='x', rotation=45)

# Average for all metals in the 6th subplot
if len(axes_flat) >= 6:
    ax = axes_flat[5]  # 6th subplot (index 5)
    avg_data = df.groupby('date')[['high', 'low']].mean().reset_index()
    ax.plot(avg_data['date'], avg_data['high'], label="Avg High", color="red", linewidth=2)
    ax.plot(avg_data['date'], avg_data['low'], label="Avg Low", color="blue", linewidth=2)
    ax.set_title("All Metals")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    
    # Format x-axis for the average plot as well
    ax.xaxis.set_major_locator(mdates.YearLocator(base=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('Average Prices of Precious Metals (2000 - 2024)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```

---

## 8) Running the notebook

1. Open the .ipynb in Jupyter.
2. Edit the data path to the appropriate location of the .csv.
3. Run cells **top‑to‑bottom**.
4. Review the outputs (tables, charts, metrics).

---
