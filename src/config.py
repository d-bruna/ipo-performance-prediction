"""
Configuration file for IPO prediction models.
"""

import os

# Paths
DATA_PATH = "data/processed/ipo_data_with_1year_excess_returns_CLEANED_FINAL.csv"
RESULTS_PATH = "results/"
FIGURES_PATH = "results/figures/"
TABLES_PATH = "results/tables/"
MODELS_PATH = "results/models/"

# Feature columns (as they appear in your CSV)
FEATURE_COLS = [
    'Sales...1.Yr.Growth', 
    'Profit.Margin', 
    'Return.on.Assets',
    'Offer.Size..M.', 
    'Shares.Outstanding..M.', 
    'Offer.Price', 
    'Market.Cap.at.Offer..M.',
    'Cash.Flow.per.Share', 
    'Instit.Owner....Shares.Out.', 
    'Instit.Owner..Shares.Held.',
    'Real.GDP.Per.Capita', 
    'OECD.Leading.Indicator', 
    'Interest.Rate',
    'Seasonally.Adjusted.Unemployment.Rate', 
    'CPI.Growth.Rate',
    'Industry.Sector', 
    'Industry.Group', 
    'Industry.Subgroup'
]

# Target columns
TARGET_1DAY_COL = '1_day_excess_return'
TARGET_1YEAR_COL = '1_Year_excess_return'

# Cross-validation settings
CV_FOLDS = 10
RANDOM_STATE = 42

# SMOTE settings
SMOTE_K_NEIGHBORS = 5
SMOTE_RANDOM_STATE = 42

# Model names
MODEL_NAMES = {
    'dt': 'Decision Tree',
    'rf': 'Random Forest',
    'gb': 'Gradient Boosting',
    'log': 'Elastic-Net Logistic'
}
