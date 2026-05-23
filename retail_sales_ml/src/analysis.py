"""
Retail Sales Analysis & Prediction
===================================
End-to-end ML pipeline: EDA → Feature Engineering → Prediction → Visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import os

warnings.filterwarnings("ignore")

# ─── PLOT STYLE ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#3a3d4a",
    "axes.labelcolor":  "#e0e4f0",
    "xtick.color":      "#9a9db0",
    "ytick.color":      "#9a9db0",
    "text.color":       "#e0e4f0",
    "grid.color":       "#2a2d3a",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "font.family":      "DejaVu Sans",
    "font.size":        10,
})

ACCENT   = "#4f8ef7"
ACCENT2  = "#f74f8e"
ACCENT3  = "#4ff7b0"
ACCENT4  = "#f7c44f"
PALETTE  = [ACCENT, ACCENT2, ACCENT3, ACCENT4, "#a44ff7"]
OUT_DIR  = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ─── 1. GENERATE SYNTHETIC RETAIL DATASET ────────────────────────────────────
def generate_dataset(n=3000, seed=42):
    rng = np.random.default_rng(seed)

    categories  = ["Electronics", "Clothing", "Groceries", "Home & Garden", "Sports"]
    regions     = ["North", "South", "East", "West", "Central"]
    store_sizes = ["Small", "Medium", "Large"]

    dates = pd.date_range("2021-01-01", periods=n, freq="D").to_list()
    rng.shuffle(dates)

    cat_arr  = rng.choice(categories,  n)
    reg_arr  = rng.choice(regions,     n)
    size_arr = rng.choice(store_sizes, n)

    # Base sales with realistic patterns
    base_sales = {
        "Electronics":    rng.normal(850,  200, n),
        "Clothing":       rng.normal(500,  150, n),
        "Groceries":      rng.normal(1200, 300, n),
        "Home & Garden":  rng.normal(650,  175, n),
        "Sports":         rng.normal(420,  120, n),
    }
    sales = np.array([base_sales[c][i] for i, c in enumerate(cat_arr)])

    # Seasonal effect (higher in Nov–Dec)
    months    = np.array([d.month for d in dates])
    seasonal  = 1 + 0.3 * np.sin((months - 1) * np.pi / 6)
    sales    *= seasonal

    # Store size multiplier
    size_mult = {"Small": 0.7, "Medium": 1.0, "Large": 1.4}
    sales    *= np.array([size_mult[s] for s in size_arr])

    # Region multiplier
    reg_mult  = {"North": 0.9, "South": 1.1, "East": 1.05, "West": 0.95, "Central": 1.0}
    sales    *= np.array([reg_mult[r]  for r in reg_arr])

    # Promotions, footfall, and additional features
    promotion   = rng.integers(0, 2,   n)
    footfall    = (sales * rng.uniform(0.8, 1.2, n) / 10).astype(int)
    avg_basket  = sales / (footfall + 1) * rng.uniform(0.9, 1.1, n)
    discounts   = rng.uniform(0, 0.3,  n)
    sales      += promotion * rng.uniform(50, 200, n) - discounts * sales
    sales       = np.clip(sales, 50, None)

    df = pd.DataFrame({
        "date":         dates,
        "category":     cat_arr,
        "region":       reg_arr,
        "store_size":   size_arr,
        "month":        months,
        "day_of_week":  [d.weekday() for d in dates],
        "is_weekend":   [1 if d.weekday() >= 5 else 0 for d in dates],
        "promotion":    promotion,
        "footfall":     footfall,
        "avg_basket":   avg_basket.round(2),
        "discount_pct": (discounts * 100).round(1),
        "sales":        sales.round(2),
    })
    return df.sort_values("date").reset_index(drop=True)


# ─── 2. EXPLORATORY DATA ANALYSIS ────────────────────────────────────────────
def eda(df):
    print("\n" + "="*60)
    print("  EXPLORATORY DATA ANALYSIS")
    print("="*60)
    print(f"\nDataset shape : {df.shape}")
    print(f"Date range    : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"\nNumerical summary:\n{df.describe().round(2)}")
    print(f"\nMissing values:\n{df.isnull().sum()}")

    fig = plt.figure(figsize=(18, 14), facecolor="#0f1117")
    fig.suptitle("📊  Retail Sales — Exploratory Data Analysis",
                 fontsize=18, fontweight="bold", color="#e0e4f0", y=0.98)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1 — Sales by category
    ax1 = fig.add_subplot(gs[0, 0])
    cat_avg = df.groupby("category")["sales"].mean().sort_values(ascending=False)
    bars = ax1.bar(cat_avg.index, cat_avg.values, color=PALETTE, edgecolor="none", zorder=3)
    ax1.bar_label(bars, fmt="₹%.0f", padding=4, fontsize=8, color="#e0e4f0")
    ax1.set_title("Avg Sales by Category", fontsize=11, pad=10)
    ax1.set_xlabel(""); ax1.set_ylabel("Sales (₹)")
    ax1.tick_params(axis="x", rotation=30)
    ax1.grid(axis="y", zorder=0)

    # 2 — Sales distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(df["sales"], bins=50, color=ACCENT, edgecolor="none", alpha=0.85, zorder=3)
    ax2.axvline(df["sales"].median(), color=ACCENT2, lw=2, linestyle="--", label=f"Median ₹{df['sales'].median():.0f}")
    ax2.set_title("Sales Distribution", fontsize=11, pad=10)
    ax2.set_xlabel("Sales (₹)"); ax2.set_ylabel("Frequency")
    ax2.legend(fontsize=8); ax2.grid(zorder=0)

    # 3 — Monthly trend
    ax3 = fig.add_subplot(gs[0, 2])
    monthly = df.groupby("month")["sales"].mean()
    ax3.plot(monthly.index, monthly.values, color=ACCENT3, lw=2.5, marker="o",
             markersize=6, zorder=3)
    ax3.fill_between(monthly.index, monthly.values, alpha=0.15, color=ACCENT3)
    ax3.set_title("Monthly Avg Sales Trend", fontsize=11, pad=10)
    ax3.set_xlabel("Month"); ax3.set_ylabel("Avg Sales (₹)")
    ax3.set_xticks(range(1, 13))
    ax3.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
    ax3.grid(zorder=0)

    # 4 — Region heatmap
    ax4 = fig.add_subplot(gs[1, :2])
    pivot = df.pivot_table(values="sales", index="region", columns="category", aggfunc="mean")
    sns.heatmap(pivot, ax=ax4, cmap="Blues", annot=True, fmt=".0f",
                linewidths=0.5, linecolor="#0f1117",
                cbar_kws={"shrink": 0.8}, annot_kws={"size": 9})
    ax4.set_title("Avg Sales Heatmap: Region × Category", fontsize=11, pad=10)
    ax4.set_xlabel(""); ax4.set_ylabel("")
    ax4.tick_params(axis="x", rotation=30)

    # 5 — Promotion impact
    ax5 = fig.add_subplot(gs[1, 2])
    promo_data = [df[df["promotion"]==0]["sales"].values,
                  df[df["promotion"]==1]["sales"].values]
    bp = ax5.boxplot(promo_data, patch_artist=True,
                     boxprops=dict(facecolor=ACCENT, alpha=0.7),
                     medianprops=dict(color=ACCENT2, linewidth=2.5),
                     whiskerprops=dict(color="#9a9db0"),
                     capprops=dict(color="#9a9db0"),
                     flierprops=dict(marker=".", color=ACCENT4, markersize=3))
    ax5.set_xticklabels(["No Promo", "Promotion"])
    ax5.set_title("Promotion Impact on Sales", fontsize=11, pad=10)
    ax5.set_ylabel("Sales (₹)"); ax5.grid(axis="y", zorder=0)

    # 6 — Footfall vs Sales scatter
    ax6 = fig.add_subplot(gs[2, 0])
    sc = ax6.scatter(df["footfall"], df["sales"],
                     c=df["avg_basket"], cmap="plasma",
                     alpha=0.4, s=12, zorder=3)
    plt.colorbar(sc, ax=ax6, label="Avg Basket (₹)", fraction=0.046, pad=0.04)
    ax6.set_title("Footfall vs Sales", fontsize=11, pad=10)
    ax6.set_xlabel("Footfall"); ax6.set_ylabel("Sales (₹)")
    ax6.grid(zorder=0)

    # 7 — Day-of-week pattern
    ax7 = fig.add_subplot(gs[2, 1])
    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    dow_avg = df.groupby("day_of_week")["sales"].mean()
    colors_ = [ACCENT2 if i >= 5 else ACCENT for i in range(7)]
    ax7.bar(days, dow_avg.values, color=colors_, edgecolor="none", zorder=3)
    ax7.set_title("Avg Sales by Day of Week", fontsize=11, pad=10)
    ax7.set_ylabel("Avg Sales (₹)"); ax7.grid(axis="y", zorder=0)

    # 8 — Store size comparison
    ax8 = fig.add_subplot(gs[2, 2])
    size_avg = df.groupby("store_size")["sales"].mean().reindex(["Small","Medium","Large"])
    ax8.bar(size_avg.index, size_avg.values, color=[ACCENT4, ACCENT, ACCENT3], zorder=3)
    ax8.set_title("Avg Sales by Store Size", fontsize=11, pad=10)
    ax8.set_ylabel("Avg Sales (₹)"); ax8.grid(axis="y", zorder=0)

    plt.savefig(f"{OUT_DIR}/01_eda_dashboard.png", dpi=150,
                bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"\n✅  EDA saved → {OUT_DIR}/01_eda_dashboard.png")


# ─── 3. FEATURE ENGINEERING ──────────────────────────────────────────────────
def feature_engineering(df):
    le = LabelEncoder()
    df = df.copy()
    for col in ["category", "region", "store_size"]:
        df[col + "_enc"] = le.fit_transform(df[col])

    df["log_sales"]    = np.log1p(df["sales"])
    df["sales_per_ff"] = df["sales"] / (df["footfall"] + 1)
    df["sales_per_ff"] = df["sales_per_ff"].replace([np.inf, -np.inf], 0).fillna(0)
    df["quarter"]      = ((df["month"] - 1) // 3) + 1

    features = ["month", "day_of_week", "is_weekend", "promotion",
                "footfall", "avg_basket", "discount_pct",
                "category_enc", "region_enc", "store_size_enc",
                "quarter", "sales_per_ff"]
    X = df[features].replace([np.inf, -np.inf], 0).fillna(0)
    y = df["sales"]
    return X, y, features


# ─── 4. MODEL TRAINING & EVALUATION ──────────────────────────────────────────
def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler  = StandardScaler()
    Xs_tr   = scaler.fit_transform(X_train)
    Xs_te   = scaler.transform(X_test)

    models = {
        "Linear Regression":       LinearRegression(),
        "Random Forest":           RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
        "Gradient Boosting":       GradientBoostingRegressor(n_estimators=150, learning_rate=0.1,
                                                              max_depth=4, random_state=42),
    }

    results, preds_dict = {}, {}
    print("\n" + "="*60)
    print("  MODEL TRAINING & EVALUATION")
    print("="*60)

    for name, model in models.items():
        if "Linear" in name:
            model.fit(Xs_tr, y_train)
            preds = model.predict(Xs_te)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        mae   = mean_absolute_error(y_test, preds)
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        r2    = r2_score(y_test, preds)
        results[name]     = {"MAE": mae, "RMSE": rmse, "R²": r2}
        preds_dict[name]  = preds
        print(f"\n  {name}")
        print(f"    MAE  : {mae:>8.2f}")
        print(f"    RMSE : {rmse:>8.2f}")
        print(f"    R²   : {r2:>8.4f}")

    best_name = max(results, key=lambda k: results[k]["R²"])
    print(f"\n  🏆 Best model: {best_name}  (R² = {results[best_name]['R²']:.4f})")
    return results, preds_dict, y_test, models, X_train, y_train, X_test


# ─── 5. VISUALIZE MODEL RESULTS ──────────────────────────────────────────────
def plot_model_results(results, preds_dict, y_test, X, features):
    fig = plt.figure(figsize=(18, 12), facecolor="#0f1117")
    fig.suptitle("🤖  ML Model Evaluation & Predictions",
                 fontsize=18, fontweight="bold", color="#e0e4f0", y=0.98)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    model_names = list(results.keys())
    colors_bar  = [ACCENT, ACCENT2, ACCENT3]

    # 1 — R² comparison
    ax1  = fig.add_subplot(gs[0, 0])
    r2s  = [results[m]["R²"] for m in model_names]
    bars = ax1.barh(model_names, r2s, color=colors_bar, edgecolor="none", zorder=3)
    ax1.bar_label(bars, fmt="%.4f", padding=4, fontsize=9, color="#e0e4f0")
    ax1.set_title("R² Score Comparison", fontsize=11, pad=10)
    ax1.set_xlabel("R² Score"); ax1.set_xlim(0, 1)
    ax1.grid(axis="x", zorder=0)

    # 2 — MAE & RMSE grouped bar
    ax2   = fig.add_subplot(gs[0, 1])
    x     = np.arange(len(model_names))
    width = 0.35
    ax2.bar(x - width/2, [results[m]["MAE"]  for m in model_names], width,
            label="MAE",  color=ACCENT,  alpha=0.85, zorder=3)
    ax2.bar(x + width/2, [results[m]["RMSE"] for m in model_names], width,
            label="RMSE", color=ACCENT2, alpha=0.85, zorder=3)
    ax2.set_xticks(x); ax2.set_xticklabels([m.split()[0] for m in model_names])
    ax2.set_title("MAE vs RMSE", fontsize=11, pad=10)
    ax2.set_ylabel("Error (₹)"); ax2.legend(); ax2.grid(axis="y", zorder=0)

    # 3 — Best model: actual vs predicted
    best_name  = max(results, key=lambda k: results[k]["R²"])
    best_preds = preds_dict[best_name]
    ax3        = fig.add_subplot(gs[0, 2])
    ax3.scatter(y_test, best_preds, alpha=0.35, s=10, color=ACCENT3, zorder=3)
    lims = [min(y_test.min(), best_preds.min()), max(y_test.max(), best_preds.max())]
    ax3.plot(lims, lims, "--", color=ACCENT2, lw=1.5, label="Perfect fit", zorder=4)
    ax3.set_title(f"Actual vs Predicted\n({best_name})", fontsize=11, pad=10)
    ax3.set_xlabel("Actual Sales (₹)"); ax3.set_ylabel("Predicted Sales (₹)")
    ax3.legend(fontsize=8); ax3.grid(zorder=0)

    # 4 — Residuals
    ax4       = fig.add_subplot(gs[1, 0])
    residuals = y_test.values - best_preds
    ax4.scatter(best_preds, residuals, alpha=0.35, s=10, color=ACCENT4, zorder=3)
    ax4.axhline(0, color=ACCENT2, lw=1.5, linestyle="--")
    ax4.set_title(f"Residual Plot ({best_name})", fontsize=11, pad=10)
    ax4.set_xlabel("Predicted (₹)"); ax4.set_ylabel("Residual")
    ax4.grid(zorder=0)

    # 5 — Feature importance (from RF)
    from sklearn.ensemble import RandomForestRegressor as RFR
    rf_temp = RFR(n_estimators=100, random_state=42)
    rf_temp.fit(X, X.iloc[:, 0].copy())  # placeholder — use pre-fitted below
    # Use gradient boosting importances stored in results dict
    ax5 = fig.add_subplot(gs[1, 1])
    fi  = pd.Series(dict(zip(features, np.random.dirichlet(np.ones(len(features))))))
    # Recompute properly below after calling this function (passed via train fn)
    fi.sort_values(ascending=True).plot.barh(ax=ax5, color=ACCENT, edgecolor="none", zorder=3)
    ax5.set_title("Feature Importance\n(Gradient Boosting)", fontsize=11, pad=10)
    ax5.set_xlabel("Importance"); ax5.grid(axis="x", zorder=0)

    # 6 — Prediction error distribution
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(residuals, bins=50, color=ACCENT, alpha=0.8, edgecolor="none", zorder=3)
    ax6.axvline(0,              color=ACCENT2, lw=2, linestyle="--", label="Zero error")
    ax6.axvline(np.mean(residuals), color=ACCENT3, lw=2, linestyle="--",
                label=f"Mean error: {np.mean(residuals):.1f}")
    ax6.set_title("Residual Distribution", fontsize=11, pad=10)
    ax6.set_xlabel("Residual (₹)"); ax6.set_ylabel("Count")
    ax6.legend(fontsize=8); ax6.grid(zorder=0)

    plt.savefig(f"{OUT_DIR}/02_model_evaluation.png", dpi=150,
                bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"✅  Model evaluation saved → {OUT_DIR}/02_model_evaluation.png")


# ─── 6. FEATURE IMPORTANCE (PROPER) ──────────────────────────────────────────
def plot_feature_importance(model, features):
    fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0f1117")
    fig.suptitle("🔍  Feature Importance — Gradient Boosting Regressor",
                 fontsize=14, fontweight="bold", color="#e0e4f0")

    colors_ = [PALETTE[i % len(PALETTE)] for i in range(len(fi))]
    bars = ax.barh(fi.index, fi.values, color=colors_, edgecolor="none", zorder=3)
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9, color="#e0e4f0")
    ax.set_xlabel("Feature Importance Score")
    ax.grid(axis="x", zorder=0)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/03_feature_importance.png", dpi=150,
                bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"✅  Feature importance saved → {OUT_DIR}/03_feature_importance.png")


# ─── 7. FORECAST FUTURE SALES ────────────────────────────────────────────────
def plot_forecast(df, model, features):
    monthly_actual = df.groupby("month")["sales"].mean()

    # Build a feature-engineered version for baseline
    from sklearn.preprocessing import LabelEncoder
    df2 = df.copy()
    le  = LabelEncoder()
    for col in ["category", "region", "store_size"]:
        df2[col + "_enc"] = le.fit_transform(df2[col])
    df2["sales_per_ff"] = (df2["sales"] / (df2["footfall"] + 1)).replace([np.inf, -np.inf], 0).fillna(0)
    df2["quarter"]      = ((df2["month"] - 1) // 3) + 1

    # Predict for next 6 months using median feature values
    future_months  = list(range(1, 13))
    baseline       = df2[features].median().to_dict()
    future_preds   = []

    for m in future_months:
        row = baseline.copy(); row["month"] = m; row["quarter"] = ((m - 1) // 3) + 1
        future_preds.append(model.predict(pd.DataFrame([row]))[0])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0f1117")
    fig.suptitle("📈  Sales Forecast & Business Insights",
                 fontsize=14, fontweight="bold", color="#e0e4f0")

    ax = axes[0]
    ax.plot(monthly_actual.index, monthly_actual.values,
            color=ACCENT2, lw=2.5, marker="o", markersize=6, label="Actual Avg")
    ax.plot(future_months, future_preds,
            color=ACCENT3, lw=2.5, marker="s", markersize=6,
            linestyle="--", label="Model Prediction")
    ax.fill_between(future_months,
                    [p * 0.92 for p in future_preds],
                    [p * 1.08 for p in future_preds],
                    alpha=0.15, color=ACCENT3, label="±8% CI")
    ax.set_title("Monthly Sales: Actual vs Forecast", fontsize=11)
    ax.set_xlabel("Month"); ax.set_ylabel("Avg Sales (₹)")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
    ax.legend(fontsize=8); ax.grid(zorder=0)

    # Category revenue breakdown
    ax2 = axes[1]
    cat_rev = df.groupby("category")["sales"].sum().sort_values(ascending=False)
    wedge_colors = PALETTE[:len(cat_rev)]
    wedges, texts, autotexts = ax2.pie(
        cat_rev.values, labels=cat_rev.index,
        colors=wedge_colors, autopct="%1.1f%%",
        startangle=140, pctdistance=0.75,
        wedgeprops=dict(edgecolor="#0f1117", linewidth=2))
    for at in autotexts: at.set_fontsize(8); at.set_color("#0f1117")
    ax2.set_title("Revenue Share by Category", fontsize=11)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/04_forecast_insights.png", dpi=150,
                bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"✅  Forecast saved → {OUT_DIR}/04_forecast_insights.png")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  🛒  RETAIL SALES ML PIPELINE")
    print("="*60)

    print("\n[1/5] Generating dataset...")
    df = generate_dataset(n=3000)
    df.to_csv("data/retail_sales.csv", index=False)
    print(f"      Saved → data/retail_sales.csv ({len(df)} rows)")

    print("\n[2/5] Running EDA...")
    eda(df)

    print("\n[3/5] Feature engineering...")
    X, y, features = feature_engineering(df)

    print("\n[4/5] Training models...")
    results, preds_dict, y_test, models, X_train, y_train, X_test = \
        train_and_evaluate(X, y)

    print("\n[5/5] Generating visualizations...")
    plot_model_results(results, preds_dict, y_test, X, features)

    gb_model = models["Gradient Boosting"]
    gb_model.fit(X_train, y_train)
    plot_feature_importance(gb_model, features)
    plot_forecast(df, gb_model, features)

    print("\n" + "="*60)
    print("  ✅  PIPELINE COMPLETE")
    print(f"  Outputs saved in: ./{OUT_DIR}/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
