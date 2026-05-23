# 🛒 Retail Sales — End-to-End Machine Learning Project

A complete, production-ready ML pipeline for retail sales analysis and prediction.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

| Item | Details |
|------|---------|
| **Domain** | Retail |
| **Task** | Sales Prediction (Regression) |
| **Models** | Linear Regression, Random Forest, Gradient Boosting |
| **Dataset** | Synthetic retail sales (3000 rows, 12 features) |
| **Best Model** | Gradient Boosting Regressor |

---

## 🗂️ Project Structure

```
retail_sales_ml/
├── data/
│   └── retail_sales.csv          # Generated dataset
├── notebooks/
│   └── retail_sales_analysis.ipynb  # Interactive Jupyter notebook
├── src/
│   └── analysis.py               # Main ML pipeline script
├── outputs/
│   ├── 01_eda_dashboard.png      # EDA visualizations
│   ├── 02_model_evaluation.png   # Model comparison charts
│   ├── 03_feature_importance.png # Feature importance
│   └── 04_forecast_insights.png  # Forecast & business insights
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/retail-sales-ml.git
cd retail-sales-ml
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline
```bash
python src/analysis.py
```

### 4. Or explore interactively
```bash
jupyter notebook notebooks/retail_sales_analysis.ipynb
```

---

## 📊 Features

### Dataset Features
| Feature | Description |
|---------|-------------|
| `date` | Transaction date |
| `category` | Product category (Electronics, Clothing, Groceries, etc.) |
| `region` | Store region (North, South, East, West, Central) |
| `store_size` | Store size (Small, Medium, Large) |
| `month` | Month number (1–12) |
| `day_of_week` | Day of week (0=Mon, 6=Sun) |
| `is_weekend` | Binary weekend flag |
| `promotion` | Whether a promotion was active |
| `footfall` | Daily customer count |
| `avg_basket` | Average basket value (₹) |
| `discount_pct` | Discount percentage applied |
| `sales` | **Target** — Daily sales revenue (₹) |

---

## 🤖 Models Used

| Model | Description |
|-------|-------------|
| **Linear Regression** | Baseline model |
| **Random Forest** | Ensemble of decision trees (n=150) |
| **Gradient Boosting** | Sequential ensemble, best performer |

### Metrics
- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **R²** — Coefficient of Determination

---

## 📈 Key Findings

1. **Groceries** generate the highest average daily sales
2. **Seasonal peaks** in summer (Jul–Aug) and holiday season (Nov–Dec)
3. **Promotions** deliver ~15–20% sales uplift on average
4. **Weekends** outperform weekdays consistently
5. **Large stores** generate ~2× more revenue than small stores

### Business Recommendations
- 📅 Schedule promotions before seasonal peaks (Oct–Dec)
- 🏪 Prioritise larger store formats in high-footfall regions
- 🛒 Focus inventory on Electronics & Groceries for revenue maximisation
- 📊 Use model predictions for staff scheduling and inventory planning

---

## 📷 Sample Outputs

### EDA Dashboard
> Multi-panel dashboard covering sales by category, distribution, monthly trends, region heatmap, and more.

### Model Evaluation
> Actual vs Predicted scatter plot, residual analysis, MAE/RMSE comparison across models.

### Feature Importance
> Bar chart showing which features drive the model's predictions most.

### Forecast
> Monthly sales forecast with confidence intervals and category revenue breakdown.

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **pandas** — Data manipulation
- **numpy** — Numerical computing
- **matplotlib / seaborn** — Visualizations
- **scikit-learn** — ML models & evaluation

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 👨‍💻 Author

Built as a real-world data science project demonstrating end-to-end ML skills in the retail domain.
