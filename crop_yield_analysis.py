# ============================================================
# Tanzania Crop Yield Intelligence — ML Prediction Study
# Author: Candy-Mellania Severin
# MSc Business Analytics & Data Science, UMCS Lublin
# Data: FAOSTAT (UN FAO) | Period: 1990–2024
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ============================================================
# STEP 1 — LOAD AND CLEAN DATA
# ============================================================

df = pd.read_csv('fao_yield_maize_rice_cassava.csv', encoding='utf-8-sig')

# Keep only the columns we need
df_clean = df[['Item', 'Year', 'Value']].copy()
df_clean.columns = ['Crop', 'Year', 'Yield_kg_ha']

# Filter to thesis crops only
thesis_crops = ['Maize (corn)', 'Rice', 'Cassava, fresh']
df_thesis = df_clean[df_clean['Crop'].isin(thesis_crops)].copy()
df_thesis = df_thesis.reset_index(drop=True)

print("=== DATASET OVERVIEW ===")
print(f"Shape: {df_thesis.shape}")
print(df_thesis['Crop'].value_counts())

# ============================================================
# STEP 2 — DESCRIPTIVE STATISTICS
# ============================================================

print("\n=== DESCRIPTIVE STATISTICS ===")
summary = df_thesis.groupby('Crop')['Yield_kg_ha'].agg(
    Count='count',
    Mean='mean',
    Std='std',
    Min='min',
    Max='max'
).round(1)
print(summary)

# ============================================================
# STEP 3 — TREND VISUALISATION
# ============================================================

crops = ['Maize (corn)', 'Rice', 'Cassava, fresh']
colors = ['darkgreen', 'goldenrod', 'sienna']
titles = ['Maize', 'Rice', 'Cassava']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (crop, color, title) in enumerate(zip(crops, colors, titles)):
    crop_data = df_thesis[df_thesis['Crop'] == crop]
    axes[i].plot(crop_data['Year'], crop_data['Yield_kg_ha'],
                 color=color, linewidth=2.5, marker='o', markersize=3)
    axes[i].set_title(title, fontsize=14, fontweight='bold')
    axes[i].set_xlabel('Year')
    axes[i].set_ylabel('Yield (kg/ha)')
    axes[i].grid(True, alpha=0.3)

plt.suptitle('Tanzania Crop Yield Trends 1990–2024',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/fig1_yield_trends.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure 1 saved.")

# ============================================================
# STEP 4 — ML MODELS
# ============================================================

future_years = np.array([[2025],[2026],[2027],[2028],[2029],[2030]])
results = []

crop_configs = [
    ('Maize (corn)', 'Maize', 'darkgreen'),
    ('Rice', 'Rice', 'goldenrod'),
    ('Cassava, fresh', 'Cassava', 'sienna')
]

for crop_key, crop_label, crop_color in crop_configs:

    crop_data = df_thesis[df_thesis['Crop'] == crop_key].copy()
    X = crop_data[['Year']].values
    Y = crop_data['Yield_kg_ha'].values

    # Build models
    lr = LinearRegression()
    dt = DecisionTreeRegressor(max_depth=4, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)

    lr.fit(X, Y)
    dt.fit(X, Y)
    rf.fit(X, Y)

    # Store results
    for model_name, model in [('Linear Regression', lr),
                                ('Decision Tree', dt),
                                ('Random Forest', rf)]:
        r2 = model.score(X, Y)
        mae = mean_absolute_error(Y, model.predict(X))
        results.append({
            'Crop': crop_label,
            'Model': model_name,
            'R²': round(r2, 3),
            'MAE (kg/ha)': round(mae, 1)
        })

    # Plot all three models
    lr_fitted = lr.predict(X)
    dt_fitted = dt.predict(X)
    rf_fitted = rf.predict(X)

    all_years = np.append(X.flatten(), future_years.flatten())
    lr_all = np.append(lr_fitted, lr.predict(future_years))
    dt_all = np.append(dt_fitted, dt.predict(future_years))
    rf_all = np.append(rf_fitted, rf.predict(future_years))

    plt.figure(figsize=(12, 6))
    plt.plot(crop_data['Year'], Y, color=crop_color, linewidth=2.5,
             marker='o', markersize=4, label='Actual Yield', zorder=5)
    plt.plot(all_years, lr_all, color='red', linewidth=1.8,
             linestyle='--', label=f'Linear Regression (R²={round(lr.score(X,Y),3)})')
    plt.plot(all_years, dt_all, color='orange', linewidth=1.8,
             linestyle='--', label=f'Decision Tree (R²={round(dt.score(X,Y),3)})')
    plt.plot(all_years, rf_all, color='steelblue', linewidth=1.8,
             linestyle='--', label=f'Random Forest (R²={round(rf.score(X,Y),3)})')
    plt.axvline(x=2024, color='gray', linestyle=':', linewidth=1.5,
                label='Forecast starts')

    if crop_label == 'Cassava':
        plt.axvspan(1990, 2003, alpha=0.08, color='red',
                    label='CMD outbreak period')

    plt.title(f'{crop_label} Yield — All Three Models Compared (1990–2030)',
              fontsize=13, fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('Yield (kg/ha)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'figures/fig_{crop_label.lower()}_models.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print(f"{crop_label} figure saved.")

# ============================================================
# STEP 5 — MASTER RESULTS TABLE
# ============================================================

print("\n=== MASTER RESULTS TABLE ===")
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
results_df.to_csv('outputs/model_results.csv', index=False)
print("\nResults saved to outputs/model_results.csv")