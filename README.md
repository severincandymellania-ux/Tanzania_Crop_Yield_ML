# Tanzania Crop Yield Intelligence — ML Prediction Study

> Predicting maize, rice and cassava yields in Tanzania using machine.
> 
> Learning to support food security policy and agricultural lending 
> 
> Risk assessment.

## Live Dashboard

🌍 [View Interactive Tableau Public Dashboard](https://public.tableau.com/views/tanzania_crop_yield_prediction/TanzaniaYieldDashboard)

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

## Phase 1 Results — Univariate Models (Year → Yield)

| Crop | Model | R² | MAE (kg/ha) |
|---|---|---|---|
| Maize | Linear Regression | 0.003 | 315.8 |
| Maize | Decision Tree | 0.960 | 57.7 |
| Maize | Random Forest | 0.844 | 125.2 |
| Rice | Linear Regression | 0.722 | 258.9 |
| Rice | Decision Tree | 0.948 | 84.5 |
| Rice | Random Forest | 0.945 | 107.2 |
| Cassava | Linear Regression | 0.437 | 1405.4 |
| Cassava | Decision Tree | 0.966 | 253.9 |
| Cassava | Random Forest | 0.941 | 403.5 |

## Phase 2 Results — Multivariate Models (Climate + Fertilizer + Area → Yield)

| Crop | Model | R² | MAE (kg/ha) |
|---|---|---|---|
| Maize | Linear Regression | 0.632 | 226.0 |
| Maize | Decision Tree | 0.919 | 88.7 |
| Maize | Random Forest | 0.872 | 126.1 |
| Rice | Linear Regression | 0.789 | 227.7 |
| Rice | Decision Tree | 0.979 | 50.2 |
| Rice | Random Forest | 0.970 | 83.7 |
| Cassava | Linear Regression | 0.825 | 728.4 |
| Cassava | Decision Tree | 0.974 | 210.2 |
| Cassava | Random Forest | 0.968 | 319.6 |

## Key Findings

**1. Random Forest is the recommended model across all crops and phases.**

**2. Climate variables dramatically improve prediction accuracy:**
- Maize Linear Regression: R² 0.003 → 0.632 (+629%)
- Cassava Linear Regression: R² 0.437 → 0.825 (+89%)

**3. Model selection has real-world consequences:**
Linear Regression predicted cassava yields would continue declining 
to 3,912 kg/ha by 2030. Random Forest correctly identified the 
post-2003 stabilisation at ~6,400 kg/ha. Wrong model selection in 
agricultural lending contexts leads to systematic underestimation 
of farmer productivity.

**4. Crop-specific findings:**
- Rice is Tanzania's most predictable food crop — driven by 
  fertilizer use (r=0.81) and rainfall (r=0.71)
- Maize yield volatility is climate-driven, not random
- Cassava shows negative correlation with minimum temperature 
  (r=-0.53) — a climate change warning signal

## Data Sources

| Dataset | Source | Period |
|---|---|---|
| Crop Yield (Maize, Rice, Cassava) | FAOSTAT — UN FAO | 1990–2024 |
| Fertilizers by Nutrient (N, P, K) | FAOSTAT — UN FAO | 1990–2023 |
| Area Harvested | FAOSTAT — UN FAO | 1990–2024 |
| Land Use (Cropland/Arable) | FAOSTAT — UN FAO | 1990–2023 |
| Pesticide Use | FAOSTAT — UN FAO | 1990–2023 |
| Climate (Rainfall, T_min, T_max, Solar) | NASA POWER | 1990–2024 |

## Project Structure

## Tools & Technologies

- **Python 3.14** — core analysis
- **pandas, numpy** — data processing
- **scikit-learn** — machine learning models
- **matplotlib, seaborn** — data visualisation
- **Tableau Public** — interactive dashboard
- **PyCharm** — development environment
- **Git/GitHub** — version control

## Project Status

- [x] Phase 1 — Univariate ML models (Year → Yield)
- [x] Phase 2 — Multivariate models (Climate + Fertilizer + Area → Yield)
- [x] Phase 3 — Tableau Public interactive dashboard
- [ ] Phase 4 — Regional yield mapping (NBS Tanzania microdata)

## Limitations & Future Work

- 34–35 data points per crop limits cross-validation reliability
- Fertilizer and climate data available only to 2023
- National level analysis only — no regional breakdown
- Future work: incorporate NBS Tanzania Agricultural Sample Survey 
  microdata for regional yield mapping at district level

## Author

**Candy Mellania Severin**
MSc Business Analytics & Data Science
Maria Curie-Skłodowska University (UMCS), Lublin, Poland

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/candymellaniaseverin)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-green)](https://candymellaniaportfolio.vercel.app)
[![Tableau](https://img.shields.io/badge/Tableau-Dashboard-orange)](https://public.tableau.com/views/tanzania_crop_yield_prediction/TanzaniaYieldDashboard)