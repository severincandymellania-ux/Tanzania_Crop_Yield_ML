# ============================================================
# Climate Scenario Analysis — Tanzania Crop Yield
# Author: Candy Mellania Severin
# Question: If rainfall drops 10% from the 10-year average,
# how much does maize yield change?
# ============================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# ── STEP 1: LOAD OUR MASTER DATASET ──
# This is the same dataset we used in Phase 2
# It has yield + climate + fertilizer + area harvested

df = pd.read_csv('outputs/tableau_master_data.csv', encoding='utf-8-sig')

print("=== MASTER DATA LOADED ===")
print(df.shape)
print(df.columns.tolist())

# ── STEP 2: DEFINE FEATURES AND TARGET ──
# Same features we used in Phase 2 multivariate models
# We train one Random Forest model per crop

features = ['Year', 'Rainfall_mm', 'Temp_Max_C', 'Temp_Min_C',
            'Solar_MJ_m2', 'Nitrogen_t', 'Phosphate_t',
            'Potash_t', 'Area_ha']

crops = ['Maize (corn)', 'Rice', 'Cassava, fresh']
crop_labels = {'Maize (corn)': 'Maize',
               'Rice': 'Rice',
               'Cassava, fresh': 'Cassava'}

# ── STEP 3: CALCULATE 10-YEAR AVERAGE RAINFALL ──
# We use 2014-2023 as our 10-year reference period
# This becomes our "normal" rainfall baseline

recent = df[df['Year'] >= 2014]
avg_rainfall = recent['Rainfall_mm'].mean()
drought_rainfall = avg_rainfall * 0.90  # 10% reduction

print("=== RAINFALL SCENARIO ===")
print(f"10-year average rainfall (2014-2023): {avg_rainfall:.1f} mm")
print(f"Drought scenario (-10%):              {drought_rainfall:.1f} mm")
print(f"Difference:                           {avg_rainfall - drought_rainfall:.1f} mm less")

# ── STEP 4: BUILD RANDOM FOREST MODELS AND RUN SCENARIOS ──
# For each crop we:
# 1. Train the Random Forest on historical data
# 2. Create a "baseline" input using 2023 values
# 3. Predict yield under normal rainfall
# 4. Predict yield under drought rainfall (-10%)
# 5. Calculate the difference

scenario_results = []

for crop_key in crops:
    crop_label = crop_labels[crop_key]
    crop_df = df[df['Crop'] == crop_key].copy()

    X = crop_df[features].values
    Y = crop_df['Yield_kg_ha'].values

    # Train Random Forest — same settings as Phase 2
    rf = RandomForestRegressor(n_estimators=100,
                               max_depth=4,
                               random_state=42)
    rf.fit(X, Y)

    # Use 2023 values as our baseline input
    # (most recent year — represents current conditions)
    baseline_input = crop_df[crop_df['Year'] == 2023][features].values

    if len(baseline_input) == 0:
        # If 2023 not available use last year
        baseline_input = crop_df[features].iloc[[-1]].values

    # Scenario 1 — Normal rainfall (baseline)
    normal_pred = rf.predict(baseline_input)[0]

    # Scenario 2 — Drought (-10% rainfall)
    drought_input = baseline_input.copy()
    # Rainfall is the 2nd column in features list
    drought_input[0][1] = drought_rainfall
    drought_pred = rf.predict(drought_input)[0]

    # Calculate impact
    yield_change    = drought_pred - normal_pred
    pct_change      = (yield_change / normal_pred) * 100

    scenario_results.append({
        'Crop':                crop_label,
        'Normal_Yield_kg_ha':  round(normal_pred, 1),
        'Drought_Yield_kg_ha': round(drought_pred, 1),
        'Yield_Change_kg_ha':  round(yield_change, 1),
        'Pct_Change':          round(pct_change, 2),
        'Scenario':            'Rainfall -10% from 10-year average'
    })

    print(f"\n{crop_label}:")
    print(f"  Normal yield:   {normal_pred:.1f} kg/ha")
    print(f"  Drought yield:  {drought_pred:.1f} kg/ha")
    print(f"  Change:         {yield_change:.1f} kg/ha ({pct_change:.1f}%)")