# Tanzania Crop Yield Intelligence — ML Prediction Study

> Predicting maize, rice and cassava yields in Tanzania using machine 
> learning to support food security policy and agricultural lending 
> risk assessment.

## Problem Statement

Tanzania's food security depends on the stable production of three 
staple crops — maize, rice and cassava. Yet yield fluctuations driven 
by climate variability, disease outbreaks and agricultural input gaps 
create significant uncertainty for farmers, policymakers and lenders. 
This project applies machine learning to 35 years of official FAO data 
to predict future crop yields and identify which models perform best 
in the Tanzanian agricultural context.

## Who This Is For

- 🌾 **Agritech companies** (Rada 360) — data-driven farming decisions
- 🏦 **Agricultural lenders** (Stanbic, Tanzania Commercial Bank) — 
  crop yield risk assessment for loan portfolios
- 🌍 **NGOs and policy bodies** (FAO Tanzania, AGRA, WFP) — food 
  security early warning
- 🏛️ **Government** (Tanzania Ministry of Agriculture) — evidence-based 
  agricultural planning

## Crops Studied

| Crop | Avg Yield (kg/ha) | Key Finding |
|---|---|---|
| Maize | 1,577.7 | High volatility — climate sensitive |
| Rice | 2,164.3 | Consistent upward trend 1990–2024 |
| Cassava | 7,262.5 | Sharp CMD-driven decline, stable post-2010 |

## Models Used

- **Linear Regression** — baseline model
- **Decision Tree** — non-linear pattern detection
- **Random Forest** — ensemble model, best overall performer

## Key Results

| Crop | Best Model | R² | MAE (kg/ha) |
|---|---|---|---|
| Maize | Random Forest | 0.844 | ±125.2 |
| Rice | Random Forest | 0.945 | ±107.2 |
| Cassava | Random Forest | 0.941 | ±403.5 |

**Random Forest is the recommended model across all three crops.**

## Key Finding — Why Model Selection Matters

Linear Regression predicted cassava yields would continue declining 
to 3,912 kg/ha by 2030 — below any yield ever recorded. Random Forest 
correctly identified the post-2003 stabilisation at ~6,400 kg/ha. 
Wrong model selection in agricultural lending contexts leads to 
systematic underestimation of farmer productivity.

## Data Sources

| Dataset | Source | Period |
|---|---|---|
| Crop Yield (Maize, Rice, Cassava) | FAOSTAT — UN FAO | 1990–2024 |
| Fertilizers by Nutrient (N, P, K) | FAOSTAT — UN FAO | 1990–2023 |
| Area Harvested | FAOSTAT — UN FAO | 1990–2024 |
| Land Use (Cropland/Arable) | FAOSTAT — UN FAO | 1990–2023 |
| Pesticide Use | FAOSTAT — UN FAO | 1990–2023 |
| Climate (Rainfall, T_min, T_max, Solar) | NASA POWER | 1990–2024 |

## Tools & Technologies

- **Python 3.14** — core analysis
- **pandas, numpy** — data processing
- **scikit-learn** — machine learning models
- **matplotlib** — data visualisation
- **Tableau Public** — interactive dashboard (Phase 3)

## Project Status

- [x] Phase 1 — Univariate ML models (Year → Yield)
- [x] Phase 2 — Multivariate models (Rainfall + Fertilizer + Area → Yield)
- [ ] Phase 3 — Tableau Public dashboard

## Live Dashboard

🌍 [View Interactive Tableau Public Dashboard](https://public.tableau.com/views/tanzania_crop_yield_prediction/TanzaniaYieldDashboard)

## Author

**Candy-Mellania Severin**
MSc Business Analytics & Data Science
Maria Curie-Skłodowska University (UMCS), Lublin, Poland
[LinkedIn](https://www.linkedin.com/in/candymellaniaseverin)
[Portfolio](https://candymellaniaportfolio.vercel.app)