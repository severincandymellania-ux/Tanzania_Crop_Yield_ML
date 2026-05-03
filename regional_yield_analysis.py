import pandas as pd

# Load the key files
aass = pd.read_csv('nbs_data/AASS2024f.csv', encoding='utf-8-sig')
masika = pd.read_csv('nbs_data/MASIKACROP.csv', encoding='utf-8-sig', low_memory=False)
vuli = pd.read_csv('nbs_data/VULICROP.csv', encoding='utf-8-sig', low_memory=False)

# Check crop IDs available
print("=== MASIKA CROP IDs ===")
print(masika['crop_id'].value_counts().head(20))

print("\n=== VULI CROP IDs ===")
print(vuli['crop_id'].value_counts().head(20))

import pandas as pd

# ============================================================
# NBS Tanzania Regional Crop Yield Analysis
# Author: Candy Mellania Severin
# Data: Annual Agricultural Sample Survey 2023/24 — NBS Tanzania
# Purpose: Extract regional yield data for Tableau map
# ============================================================

# ── STEP 1: LOAD ALL FILES ──
# AASS2024f = main survey file that tells us which REGION each farm is in
# MASIKACROP = crop records for the long rainy season (Masika)
# VULICROP = crop records for the short rainy season (Vuli)

aass   = pd.read_csv('nbs_data/AASS2024f.csv', encoding='utf-8-sig')
masika = pd.read_csv('nbs_data/MASIKACROP.csv', encoding='utf-8-sig', low_memory=False)
vuli   = pd.read_csv('nbs_data/VULICROP.csv',   encoding='utf-8-sig', low_memory=False)

print("=== FILES LOADED ===")
print(f"AASS (regions):  {aass.shape}")
print(f"Masika crops:    {masika.shape}")
print(f"Vuli crops:      {vuli.shape}")

# ── STEP 2: KEEP ONLY WHAT WE NEED FROM EACH FILE ──
# From AASS we only need: holding_id (farm ID) and region
# We also keep 'weight' — this is a survey weight that makes
# the sample representative of all Tanzania (standard in survey data)

aass_clean = aass[['holding_id', 'region', 'weight']].dropna(subset=['region'])

print(f"\n=== REGIONS AVAILABLE ===")
print(f"Total regions: {aass_clean['region'].nunique()}")
print(aass_clean['region'].unique())

# ── STEP 3: FILTER TO THESIS CROPS ONLY ──
# Maize, Paddy (rice) and Cassava
# We label Paddy as Rice so it matches our FAO dataset naming

thesis_crops = ['Maize', 'Paddy', 'Cassava']

masika_filtered = masika[masika['crop_id'].isin(thesis_crops)].copy()
vuli_filtered   = vuli[vuli['crop_id'].isin(thesis_crops)].copy()

# Rename Paddy to Rice for consistency
masika_filtered['crop_id'] = masika_filtered['crop_id'].replace('Paddy', 'Rice')
vuli_filtered['crop_id']   = vuli_filtered['crop_id'].replace('Paddy', 'Rice')

# Add season label so we know where each record came from
masika_filtered['season'] = 'Masika'
vuli_filtered['season']   = 'Vuli'

print(f"\n=== FILTERED CROP RECORDS ===")
print(f"Masika thesis crops: {masika_filtered.shape[0]} records")
print(f"Vuli thesis crops:   {vuli_filtered.shape[0]} records")
print(f"\nCrop breakdown:")
print(masika_filtered['crop_id'].value_counts())

# ── STEP 4: COMBINE BOTH SEASONS ──
# Stack Masika and Vuli records into one combined dataset
# This gives us the fullest picture of crop production across Tanzania

all_crops = pd.concat([masika_filtered, vuli_filtered], ignore_index=True)

print(f"=== COMBINED SEASONS ===")
print(f"Total crop records: {all_crops.shape[0]}")

# ── STEP 5: LINK CROPS TO REGIONS ──
# We merge crop records with the AASS file using holding_id
# Think of it like a VLOOKUP in Excel —
# "for each crop record, find which region that farm belongs to"

all_crops_regional = all_crops.merge(
    aass_clean[['holding_id', 'region', 'weight']],
    on='holding_id',
    how='left'
)

# Check how many records got a region successfully
matched = all_crops_regional['region'].notna().sum()
total   = len(all_crops_regional)
print(f"Records matched to a region: {matched} / {total}")
print(f"Match rate: {round(matched/total*100, 1)}%")

# ── STEP 6: EXPLORE THE YIELD COLUMN ──
# We need to find which column contains the actual yield data
# Let's check what columns are available in the crop files

print(f"\n=== AVAILABLE COLUMNS IN CROP DATA ===")
print(all_crops_regional.columns.tolist())

# ── SEE ALL COLUMNS FULLY ──
for col in all_crops_regional.columns.tolist():
    print(col)

# ── STEP 7: CALCULATE YIELD PER HECTARE ──
# Yield (kg/ha) = (harvest in tonnes / area harvested in hectares) x 1000
# We use areaHarvested not areaPlanted because some planted area
# may not have been harvested due to drought, disease or other factors
# areaHarvested is the more accurate denominator for yield calculation

# First let's check these columns
print("=== YIELD COLUMN OVERVIEW ===")
print(f"harvest_ton — sample values:")
print(all_crops_regional['harvest_ton'].describe())
print(f"\nareaHarvested — sample values:")
print(all_crops_regional['areaHarvested'].describe())

# Calculate yield in kg/ha
# Remove rows where areaHarvested is zero or missing (can't divide by zero)
df_yield = all_crops_regional[
    (all_crops_regional['areaHarvested'] > 0) &
    (all_crops_regional['harvest_ton'].notna()) &
    (all_crops_regional['region'].notna())
].copy()

df_yield['yield_kg_ha'] = (
    df_yield['harvest_ton'] / df_yield['areaHarvested']
) * 1000

print(f"\n=== YIELD CALCULATION ===")
print(f"Records with valid yield: {len(df_yield)}")
print(f"\nYield summary by crop:")
print(df_yield.groupby('crop_id')['yield_kg_ha'].describe().round(1))

# ── STEP 8: AGGREGATE BY REGION AND CROP ──
# We now group by region and crop to get the average yield per region
# We use weighted average because the survey uses sampling weights
# (some farms represent more households than others in the sample)
# This makes our regional averages representative of all Tanzania

# Remove outliers — yields above 15,000 kg/ha are likely data entry errors
df_yield_clean = df_yield[df_yield['yield_kg_ha'] <= 15000].copy()

# Calculate weighted average yield per region per crop
def weighted_avg(group):
    weights = group['weight']
    values  = group['yield_kg_ha']
    # Only calculate if we have valid weights
    if weights.sum() == 0:
        return values.mean()
    return (values * weights).sum() / weights.sum()

regional_yield = df_yield_clean.groupby(
    ['region', 'crop_id']
).apply(weighted_avg).reset_index()

regional_yield.columns = ['Region', 'Crop', 'Avg_Yield_kg_ha']
regional_yield['Avg_Yield_kg_ha'] = regional_yield['Avg_Yield_kg_ha'].round(1)
regional_yield['Data_Source'] = 'NBS Tanzania AASS 2023/24'
regional_yield['Year'] = 2024

print("=== REGIONAL YIELD TABLE ===")
print(f"Shape: {regional_yield.shape}")
print(regional_yield.sort_values(['Crop', 'Avg_Yield_kg_ha'],
                                   ascending=[True, False]).to_string(index=False))

# ── STEP 9: EXPORT FOR TABLEAU MAP ──
# Save the regional yield table as a CSV for Tableau
# Tableau will use the Region column to match Tanzania's map regions

# Also add coordinates for each region as a backup
# (Tableau can geocode Tanzanian regions automatically)

regional_yield.to_csv('outputs/tableau_regional_yield.csv', index=False)
print("✅ Regional yield data exported!")
print(f"\nFile saved to: outputs/tableau_regional_yield.csv")
print(f"Rows: {len(regional_yield)}")
print(f"Regions covered: {regional_yield['Region'].nunique()}")
print(f"Crops covered: {regional_yield['Crop'].nunique()}")

# Also save a summary for the thesis
summary = regional_yield.groupby('Crop').agg(
    Regions=('Region', 'count'),
    Avg_National=('Avg_Yield_kg_ha', 'mean'),
    Max_Yield=('Avg_Yield_kg_ha', 'max'),
    Min_Yield=('Avg_Yield_kg_ha', 'min'),
    Top_Region=('Region', lambda x: regional_yield.loc[x.index, 'Avg_Yield_kg_ha'].idxmax()),
).round(1)

print("\n=== NATIONAL SUMMARY BY CROP ===")
print(summary)

# ── STEP 10: ADD MANUAL COORDINATES FOR EACH REGION ──
# Tableau's geocoding doesn't recognise Tanzanian regions
# So we provide latitude and longitude for each region manually
# These are the approximate centroids of each region

region_coords = {
    'Arusha':            (-3.3869,  36.6830),
    'Dar es Salaam':     (-6.7924,  39.2083),
    'Dodoma':            (-6.1630,  35.7516),
    'Geita':             (-2.8649,  32.1654),
    'Iringa':            (-7.7701,  35.6939),
    'Kagera':            (-1.2986,  31.2697),
    'Kaskazini Pemba':   (-5.0332,  39.7748),
    'Kaskazini Unguja':  (-5.7748,  39.3467),
    'Katavi':            (-6.8316,  31.3132),
    'Kigoma':            (-4.8770,  29.6270),
    'Kilimanjaro':       (-3.3390,  37.3390),
    'Kusini Pemba':      (-5.2897,  39.7120),
    'Kusini Unguja':     (-6.2462,  39.4425),
    'Lindi':             (-9.9982,  39.7142),
    'Manyara':           (-4.3155,  36.2540),
    'Mara':              (-1.7554,  34.0069),
    'Mbeya':             (-8.9000,  33.4600),
    'Mjini Magharibi':   (-6.1659,  39.1989),
    'Morogoro':          (-6.8242,  37.6611),
    'Mtwara':            (-10.2673, 40.1877),
    'Mwanza':            (-2.5164,  32.9175),
    'Njombe':            (-9.3333,  34.7667),
    'Pwani':             (-7.0048,  38.6529),
    'Rukwa':             (-7.9338,  31.3997),
    'Ruvuma':            (-10.6896, 35.6464),
    'Shinyanga':         (-3.6601,  33.4230),
    'Simiyu':            (-2.8370,  34.1520),
    'Singida':           (-4.8185,  34.7500),
    'Songwe':            (-8.9701,  32.6988),
    'Tabora':            (-5.0160,  32.8005),
    'Tanga':             (-5.0693,  38.9870),
}

# Add coordinates to our regional yield table
regional_yield['Latitude']  = regional_yield['Region'].map(
    lambda r: region_coords.get(r, (None, None))[0]
)
regional_yield['Longitude'] = regional_yield['Region'].map(
    lambda r: region_coords.get(r, (None, None))[1]
)

# Save updated file with coordinates
regional_yield.to_csv('outputs/tableau_regional_yield.csv', index=False)
print("✅ Regional yield data updated with coordinates!")
print(regional_yield[['Region','Crop','Avg_Yield_kg_ha',
                        'Latitude','Longitude']].head(10))