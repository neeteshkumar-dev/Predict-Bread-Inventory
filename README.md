# 🍞 Predict Bread Inventory Before It Runs Out

> AI-powered inventory forecasting for Grupo Bimbo's supply chain — predicting exactly how many bread units each shop needs before stock runs out, trained on 74 million real delivery records.

---

## 🔗 Live Pages

| Page | Description |
|------|-------------|
| Home | Project overview, key stats, model summary |
| About | Supply chain problem, why it matters, tech stack |
| Data Fields | Full data dictionary — all 10 input fields explained |
| ML Pipeline | Step-by-step model build from raw data to predictions |
| Live Predictor | Interactive tool — enter shop data, get instant prediction |

---

## 📌 What This Project Does

Grupo Bimbo — Mexico's largest bakery — delivers fresh bread, cakes, and snacks to thousands of shops every week via fixed truck routes. The core business problem is simple but expensive:

- Too much stock → bread goes stale, gets returned, wastes truck capacity  
- Too little stock → shelves go empty, sales are lost  

This project builds a **Random Forest model** that predicts the exact number of units (`Demanda_uni_equil`) each shop needs next week, using only this week's delivery data as input.

---

## 📊 Dataset

**Source:** Grupo Bimbo Inventory Demand — Kaggle Competition  

| Stat | Value |
|------|------|
| Total raw records | 74,188,840 |
| Rows used for training | 240,000 |
| Rows used for validation | 60,000 |
| Test predictions generated | 6,999,251 |
| Input features | 10 |
| Target variable | `Demanda_uni_equil` |

---

## 📥 Input Fields

| Field | Spanish Name | Type | Role |
|------|-------------|------|------|
| Week Number | `Semana` | Categorical | Context |
| Sales Depot ID | `Agencia_ID` | Categorical | Context |
| Sales Channel | `Canal_ID` | Categorical | Context |
| Delivery Route | `Ruta_SAK` | Categorical | Context |
| Shop / Client ID | `Cliente_ID` | Categorical | Identifier |
| Product ID | `Producto_ID` | Categorical | Identifier |
| Units Sold This Week | `Venta_uni_hoy` | Float32 | **97% importance** |
| Revenue This Week | `Venta_hoy` | Float32 | Signal |
| Returns Next Week (Units) | `Dev_uni_proxima` | Float32 | Signal (2%) |
| Returns Next Week (Pesos) | `Dev_proxima` | Float32 | Signal |

---

## 🌲 Model

**Algorithm:** `RandomForestRegressor` (scikit-learn)

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=30,
    max_depth=10,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)
---

**## ⚙️ Key Design Decisions

- **Log transform (`log1p`)**  
  Demand is highly skewed (many shops need 1–5 units, few need 500+). Log transformation stabilizes learning and aligns with RMSLE evaluation metric.

- **Memory optimization**  
  Using `uint8`, `uint16`, `float32` instead of default types reduces RAM usage by ~60%, making it runnable on Google Colab free tier.

- **Ensemble learning**  
  30 decision trees trained on random subsets of data → final prediction is averaged for better stability and generalization.

```
## ⚙️ ML Pipeline

1. Download     → kaggle competitions download -c grupo-bimbo-inventory-demand
2. Load         → pd.read_csv('train.csv', dtype=dtype_map, nrows=300000)
3. Transform    → df['log_demand'] = np.log1p(df['Demanda_uni_equil'])
4. Split        → train (240K rows) / validation (60K rows)
5. Train        → RandomForestRegressor(n_estimators=30, max_depth=10, n_jobs=-1)
6. Evaluate     → RMSLE on 60K held-out records
7. Predict      → 6,999,251 predictions on test.csv
8. Submit       → submission.csv → Kaggle upload
```
```
### Evaluation Metric — RMSLE
```python
from sklearn.metrics import mean\_squared\_log\_error
import numpy as np
y\_pred\_log = model.predict(X\_val)
y\_pred     = np.expm1(y\_pred\_log).clip(min=0)
y\_true     = np.expm1(y\_val)
rmsle = np.sqrt(mean\_squared\_log\_error(y\_true, y\_pred))
print(f"Validation RMSLE: {rmsle:.4f}")
```

RMSLE was chosen because it is the official Kaggle competition metric and penalises under-predictions more heavily than over-predictions — appropriate for an inventory context where running out of stock is costlier than a small overstock.

\---


## 🗂️ Project Structure

```
📦 predict-bread-inventory/
├── index.html          # Home — project overview \& stats
├── about.html          # Supply chain context \& tech stack
├── data.html           # Data dictionary — all 10 fields
├── pipeline.html       # ML pipeline walkthrough with code
├── predictor.html      # Interactive live predictor tool
└── README.md           # This file
```

\---

## 🔧 Tech Stack

|Tool|Purpose|
|-|-|
|Python 3|Core language|
|scikit-learn|Random Forest model|
|pandas|Data loading \& processing|
|NumPy|Numerical operations \& transforms|
|Google Colab|Free GPU/CPU runtime|
|Kaggle API|Dataset download|
|HTML / CSS / JS|Front-end web pages|

\---

## 🚀 Running Locally

**1. Clone the repo**

```bash
git clone https://github.com/your-username/predict-bread-inventory.git
cd predict-bread-inventory
```


**2. Install dependencies**

```bash
pip install scikit-learn pandas numpy kaggle
```

**3. Download the dataset**

```bash
# Set up your Kaggle API key first: https://www.kaggle.com/docs/api
kaggle competitions download -c grupo-bimbo-inventory-demand
unzip grupo-bimbo-inventory-demand.zip
```

**4. Run the pipeline**

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean\_squared\_log\_error

# Memory-efficient dtypes
dtype\_map = {
    'Semana': np.uint8, 'Agencia\_ID': np.uint16, 'Canal\_ID': np.uint8,
    'Ruta\_SAK': np.uint32, 'Cliente\_ID': np.uint32, 'Producto\_ID': np.uint32,
    'Venta\_uni\_hoy': np.float32, 'Venta\_hoy': np.float32,
    'Dev\_uni\_proxima': np.float32, 'Dev\_proxima': np.float32,
    'Demanda\_uni\_equil': np.float32
}

df = pd.read\_csv('train.csv', dtype=dtype\_map, nrows=300\_000)
df\['log\_demand'] = np.log1p(df\['Demanda\_uni\_equil'])

features = \['Semana','Agencia\_ID','Canal\_ID','Ruta\_SAK',
            'Cliente\_ID','Producto\_ID','Venta\_uni\_hoy',
            'Venta\_hoy','Dev\_uni\_proxima','Dev\_proxima']

train, val = df\[:240\_000], df\[240\_000:]
X\_train, y\_train = train\[features], train\['log\_demand']
X\_val,   y\_val   = val\[features],   val\['log\_demand']

model = RandomForestRegressor(n\_estimators=30, max\_depth=10,
                               n\_jobs=-1, random\_state=42)
model.fit(X\_train, y\_train)

y\_pred = np.expm1(model.predict(X\_val)).clip(min=0)
y\_true = np.expm1(y\_val)
rmsle  = np.sqrt(mean\_squared\_log\_error(y\_true, y\_pred))
print(f"Validation RMSLE: {rmsle:.4f}")
```

**5. Open the web pages**

```bash
# Just open index.html in your browser — no server needed
open index.html        # macOS
start index.html       # Windows
xdg-open index.html    # Linux
```

\---

## 📈 Results

|Metric|Value|
|-|-|
|Training rows|240,000|
|Validation rows|60,000|
|Test predictions|6,999,251|
|Top feature|`Venta\_uni\_hoy` (97% importance)|
|Evaluation metric|RMSLE (Kaggle official)|

\---

## 💡 Key Insights

* **Units sold this week alone drives 97% of next week's prediction.** This makes intuitive sense — a shop that sold 20 units this week almost certainly needs \~20 units next week.
* **Returns are the second signal (2%).** High returns = overstocked last time = send less next time.
* **Log transform is essential.** Raw demand values crash model learning due to extreme skew. `log1p` normalises the distribution and aligns predictions with how Kaggle scores them.
* **60% RAM reduction** from dtype optimisation is the difference between the pipeline running and the Colab runtime crashing.

\---

## 🏆 Acknowledgements

* [Grupo Bimbo Inventory Demand — Kaggle Competition](https://www.kaggle.com/c/grupo-bimbo-inventory-demand)
* Grupo Bimbo for releasing one of the richest public supply-chain datasets available

