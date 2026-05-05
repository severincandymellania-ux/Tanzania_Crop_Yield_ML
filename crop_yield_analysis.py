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

# ============================================================
# PHASE 2 — LOAD ADDITIONAL DATASETS
# ============================================================

# Load fertilizers
df_fert = pd.read_csv('fao_fertilizers.csv', encoding='utf-8-sig')
df_fert = df_fert[['Year', 'Item', 'Value']].copy()
print("=== FERTILIZERS ===")
print(df_fert['Item'].unique())
print(df_fert.shape)

# Load area harvested
df_area = pd.read_csv('fao_area_harvested.csv', encoding='utf-8-sig')
df_area = df_area[['Item', 'Year', 'Value']].copy()
print("\n=== AREA HARVESTED ===")
print(df_area['Item'].unique())
print(df_area.shape)

# Load NASA climate
df_climate = pd.read_csv('nasa_climate_tanzania.csv', skiprows=13)
print("\n=== NASA CLIMATE ===")
print(df_climate.head())
print(df_climate.shape)
# ============================================================
# PHASE 2 — CLEAN AND RESTRUCTURE NASA CLIMATE DATA
# ============================================================

# Read NASA file properly
nasa_raw = pd.read_csv('nasa_climate_tanzania.csv', skiprows=13)

# Fix column names
nasa_raw.columns = ['Parameter', 'Year',
                    'Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec','Annual']

# Keep only annual values and the 4 parameters we need
nasa_annual = nasa_raw[['Parameter', 'Year', 'Annual']].copy()

# Pivot so each parameter becomes a column
nasa_pivot = nasa_annual.pivot(index='Year',
                                columns='Parameter',
                                values='Annual').reset_index()

# Rename columns to something readable
nasa_pivot.columns.name = None
nasa_pivot = nasa_pivot.rename(columns={
    'ALLSKY_SFC_SW_DWN': 'Solar_MJ_m2',
    'PRECTOTCORR_SUM':   'Rainfall_mm',
    'T2M_MAX':           'Temp_Max_C',
    'T2M_MIN':           'Temp_Min_C'
})

# Filter to our study period 1990-2023
nasa_pivot = nasa_pivot[nasa_pivot['Year'] <= 2023]

print("=== NASA CLIMATE CLEANED ===")
print(nasa_pivot.shape)
print(nasa_pivot.head())
print(nasa_pivot.columns.tolist())

# ============================================================
# PHASE 2 — FIX MISSING VALUE AND PREPARE ALL DATASETS
# ============================================================

# Fill missing Solar value for 1990 with column mean
nasa_pivot['Solar_MJ_m2'] = nasa_pivot['Solar_MJ_m2'].fillna(
    nasa_pivot['Solar_MJ_m2'].mean()
)

# ── Clean fertilizers — pivot so N, P, K are columns ──
df_fert_pivot = df_fert.pivot(index='Year',
                               columns='Item',
                               values='Value').reset_index()
df_fert_pivot.columns.name = None
df_fert_pivot = df_fert_pivot.rename(columns={
    'Nutrient nitrogen N (total)':    'Nitrogen_t',
    'Nutrient phosphate P2O5 (total)':'Phosphate_t',
    'Nutrient potash K2O (total)':    'Potash_t'
})

print("=== FERTILIZERS CLEANED ===")
print(df_fert_pivot.head())

# ── Clean area harvested — one row per crop per year ──
df_area.columns = ['Crop', 'Year', 'Area_ha']

print("\n=== AREA HARVESTED CLEANED ===")
print(df_area.head())

# ── Verify all year ranges ──
print("\n=== YEAR RANGES ===")
print(f"Climate:    {nasa_pivot['Year'].min()} – {nasa_pivot['Year'].max()}")
print(f"Fertilizer: {df_fert_pivot['Year'].min()} – {df_fert_pivot['Year'].max()}")
print(f"Area:       {df_area['Year'].min()} – {df_area['Year'].max()}")
print(f"Yield:      {df_thesis['Year'].min()} – {df_thesis['Year'].max()}")

# ============================================================
# PHASE 2 — MERGE ALL DATASETS INTO MASTER TABLE
# ============================================================

# Filter yield to 1990-2023 to match other datasets
df_yield = df_thesis[df_thesis['Year'] <= 2023].copy()

# Merge yield with climate
df_master = df_yield.merge(nasa_pivot, on='Year', how='left')

# Merge with fertilizers
df_master = df_master.merge(df_fert_pivot, on='Year', how='left')

# Merge with area harvested (crop specific)
df_master = df_master.merge(df_area, on=['Crop', 'Year'], how='left')

# Check the result
print("=== MASTER TABLE ===")
print(f"Shape: {df_master.shape}")
print(df_master.head(10))
print("\nColumns:", df_master.columns.tolist())

# Check for missing values
print("\n=== MISSING VALUES ===")
print(df_master.isnull().sum())

# ============================================================
# PHASE 2 — CORRELATION HEATMAP
# ============================================================

import seaborn as sns

#calculate correlations for each crop separately
fig, axes = plt.subplots(1,3, figsize=(18, 6))

crop_labels = [
    ('Maize (corn)', 'Maize'),
    ('Rice', 'Rice'),
    ('Cassava, fresh', 'Cassava'),
]

for i, (crop_key, crop_label) in enumerate(crop_labels):
    crop_df = df_master[df_master['Crop'] == crop_key].copy()

    # Select numeric columns only
    numeric_cols = ['Yield_kg_ha', 'Rainfall_mm', 'Temp_Max_C',
                    'Temp_Min_C', 'Solar_MJ_m2', 'Nitrogen_t',
                    'Phosphate_t','Potash_t', 'Area_ha']

    corr = crop_df[numeric_cols].corr()

    sns.heatmap(corr,
                annot=True,
                fmt='.2f',
                cmap='RdYlGn',
                center=0,
                ax=axes[i],
                annot_kws={'size': 7},)

    axes[i].set_title(f'{crop_label} — Correlation Matrix',
                      fontsize=12, fontweight='bold')
    axes[i].tick_params(axis='x', rotation=45, labelsize=8)
    axes[i].tick_params(axis='y', rotation=0, labelsize=8)

plt.suptitle('Variable Correlations with Crop Yield — Tanzania 1990–2023',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/fig_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("Correlation heatmap saved.")

# ============================================================
# PHASE 2 — MULTIVARIATE ML MODELS
# ============================================================

from sklearn.model_selection import cross_val_score

# Define features for multivariate models
features = ['Year', 'Rainfall_mm', 'Temp_Max_C', 'Temp_Min_C',
            'Solar_MJ_m2', 'Nitrogen_t', 'Phosphate_t',
            'Potash_t', 'Area_ha']

results_mv = []
future_climate = None  # We'll handle forecasting separately

crop_configs = [
    ('Maize (corn)', 'Maize', 'darkgreen'),
    ('Rice', 'Rice', 'goldenrod'),
    ('Cassava, fresh', 'Cassava', 'sienna')
]

for crop_key, crop_label, crop_color in crop_configs:
    crop_df = df_master[df_master['Crop'] == crop_key].copy()

    X_mv = crop_df[features].values
    Y_mv = crop_df['Yield_kg_ha'].values

    # Build models
    lr_mv = LinearRegression()
    dt_mv = DecisionTreeRegressor(max_depth=4, random_state=42)
    rf_mv = RandomForestRegressor(n_estimators=100, max_depth=4,
                                  random_state=42)

    for model_name, model in [('Linear Regression', lr_mv),
                              ('Decision Tree', dt_mv),
                              ('Random Forest', rf_mv)]:
        model.fit(X_mv, Y_mv)
        r2 = model.score(X_mv, Y_mv)
        mae = mean_absolute_error(Y_mv, model.predict(X_mv))
        results_mv.append({
            'Crop': crop_label,
            'Model': model_name,
            'R² (Phase 2)': round(r2, 3),
            'MAE (kg/ha)': round(mae, 1)
        })

    print(f"✅ {crop_label} multivariate models done")

# Print results
print("\n=== PHASE 2 — MULTIVARIATE MODEL RESULTS ===")
results_mv_df = pd.DataFrame(results_mv)
print(results_mv_df.to_string(index=False))
results_mv_df.to_csv('outputs/model_results_phase2.csv', index=False)
print("\nResults saved to outputs/model_results_phase2.csv")

# ============================================================
# PHASE 2 — COMPARISON CHART: PHASE 1 VS PHASE 2
# ============================================================

phase1_r2 = {
    'Maize':   {'Linear Regression': 0.003, 'Decision Tree': 0.960, 'Random Forest': 0.844},
    'Rice':    {'Linear Regression': 0.722, 'Decision Tree': 0.948, 'Random Forest': 0.945},
    'Cassava': {'Linear Regression': 0.437, 'Decision Tree': 0.966, 'Random Forest': 0.941}
}

phase2_r2 = {
    'Maize':   {'Linear Regression': 0.632, 'Decision Tree': 0.919, 'Random Forest': 0.872},
    'Rice':    {'Linear Regression': 0.789, 'Decision Tree': 0.979, 'Random Forest': 0.970},
    'Cassava': {'Linear Regression': 0.825, 'Decision Tree': 0.974, 'Random Forest': 0.968}
}

crops    = ['Maize', 'Rice', 'Cassava']
models   = ['Linear Regression', 'Decision Tree', 'Random Forest']
colors_p1 = ['#AED6AE', '#FFD980', '#D4A57A']
colors_p2 = ['darkgreen', 'goldenrod', 'sienna']

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

for i, crop in enumerate(crops):
    x = np.arange(len(models))
    width = 0.35

    p1_vals = [phase1_r2[crop][m] for m in models]
    p2_vals = [phase2_r2[crop][m] for m in models]

    axes[i].bar(x - width/2, p1_vals, width,
                label='Phase 1 (Year only)',
                color=colors_p1[i], edgecolor='white')
    axes[i].bar(x + width/2, p2_vals, width,
                label='Phase 2 (Multivariate)',
                color=colors_p2[i], edgecolor='white')

    axes[i].set_title(crop, fontsize=13, fontweight='bold')
    axes[i].set_xticks(x)
    axes[i].set_xticklabels(['Linear\nReg', 'Decision\nTree', 'Random\nForest'],
                             fontsize=9)
    axes[i].set_ylabel('R² Score')
    axes[i].set_ylim(0, 1.1)
    axes[i].legend(fontsize=8)
    axes[i].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for rect, val in zip(axes[i].patches, p1_vals + p2_vals):
        axes[i].text(rect.get_x() + rect.get_width()/2,
                     rect.get_height() + 0.02,
                     f'{val:.2f}', ha='center', va='bottom',
                     fontsize=8, fontweight='bold')

plt.suptitle('Phase 1 vs Phase 2 — R² Score Comparison\nUnivariate vs Multivariate Models',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig_phase1_vs_phase2.png', dpi=150, bbox_inches='tight')
plt.show()
print("Phase comparison chart saved.")

# ============================================================
# PHASE 3 — EXPORT DATA FOR TABLEAU DASHBOARD
# ============================================================

# Export 1 — Master dataset (historical + all variables)
df_master.to_csv('outputs/tableau_master_data.csv', index=False)
print("✅ Master dataset exported for Tableau")

# Export 2 — Phase 1 model results
results_df['Phase'] = 'Phase 1 — Univariate'
results_mv_df['Phase'] = 'Phase 2 — Multivariate'

# Rename Phase 2 column to match
results_mv_df = results_mv_df.rename(columns={'R² (Phase 2)': 'R²'})

# Combine both phases
all_results = pd.concat([results_df, results_mv_df], ignore_index=True)
all_results.to_csv('outputs/tableau_model_results.csv', index=False)
print("✅ Model results exported for Tableau")

# Export 3 — Predictions table
predictions_data = []
future_yrs = [2025, 2026, 2027, 2028, 2029, 2030]

crop_configs_pred = [
    ('Maize (corn)', 'Maize'),
    ('Rice', 'Rice'),
    ('Cassava, fresh', 'Cassava')
]

for crop_key, crop_label in crop_configs_pred:
    crop_df = df_master[df_master['Crop'] == crop_key].copy()
    X_mv = crop_df[features].values
    Y_mv = crop_df['Yield_kg_ha'].values

    # Refit models
    lr_p = LinearRegression()
    dt_p = DecisionTreeRegressor(max_depth=4, random_state=42)
    rf_p = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)

    lr_p.fit(X_mv, Y_mv)
    dt_p.fit(X_mv, Y_mv)
    rf_p.fit(X_mv, Y_mv)

    # Use 2023 climate/fertilizer values as proxy for future
    last_row = crop_df[features].iloc[-1].values

    for yr in future_yrs:
        future_input = last_row.copy()
        future_input[0] = yr  # Update year
        future_input = future_input.reshape(1, -1)

        predictions_data.append({
            'Crop': crop_label,
            'Year': yr,
            'Type': 'Forecast',
            'Linear_Regression': round(float(lr_p.predict(future_input)[0]), 1),
            'Decision_Tree': round(float(dt_p.predict(future_input)[0]), 1),
            'Random_Forest': round(float(rf_p.predict(future_input)[0]), 1)
        })

    # Also add historical actuals
    for _, row in crop_df.iterrows():
        predictions_data.append({
            'Crop': crop_label,
            'Year': int(row['Year']),
            'Type': 'Actual',
            'Linear_Regression': None,
            'Decision_Tree': None,
            'Random_Forest': None,
            'Actual_Yield': row['Yield_kg_ha']
        })

predictions_df = pd.DataFrame(predictions_data)
predictions_df.to_csv('outputs/tableau_predictions.csv', index=False)
print("✅ Predictions exported for Tableau")

print("\n=== ALL TABLEAU FILES READY ===")
print("outputs/tableau_master_data.csv")
print("outputs/tableau_model_results.csv")
print("outputs/tableau_predictions.csv")
