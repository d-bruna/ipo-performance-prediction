# Predicting IPO Performance Across Investment Horizons

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains the complete code and analysis for our research paper: **"How does the importance of pre-IPO attributes change across different investment horizons when classifying IPO performance as under- or overperforming the S&P 500"**

## Authors
- **Daniel Garcia** - 637987db@eur.nl
- **Charles Pearce** - 695309cp@eur.nl  
- **Lars Stomps** - 678137ls@eur.nl
- **Johnny Hu** - 677926jh@eur.nl

*Erasmus University Rotterdam*

---

## Abstract

This paper evaluates 881 IPOs with 18 features to predict whether they outperform the S&P 500 at 1-day and 1-year horizons. We employ four machine learning classifiers and find that feature importance shifts dramatically across horizons: 1-day outcomes align with offer mechanics and macroeconomic conditions, while 1-year outcomes emphasize institutional ownership, profitability, and sales growth.

## Key Findings

- **Best 1-day model:** Gradient Boosting (70.26% accuracy, 0.654 ROC-AUC)
- **Best 1-year model:** Random Forest (65.37% accuracy, 0.691 ROC-AUC)
- **Universal predictor:** Institutional ownership is the only feature consistently important across both horizons
- **Temporal shift:** Offer Price importance drops from rank 1 (1-day) to rank 8 (1-year)
- **Profitability matters long-term:** Profit Margin rises from rank 6 (1-day) to rank 2 (1-year)

---

## Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ipo-performance-prediction.git
cd ipo-performance-prediction

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis
```bash
# Train all models and generate results
python scripts/train_models.py
```

---

## Repository Structure
```
├── data/
│   ├── raw/                  # Original data sources
│   └── processed/            # Cleaned final dataset
│       └── ipo_data_with_1year_excess_returns_CLEANED_FINAL.csv
│
├── notebooks/                # Jupyter notebooks for data pipeline
│   ├── 01_data_scrape.ipynb           # Scrape Yahoo Finance data
│   ├── 02_data_cleaning.ipynb         # Initial data cleaning
│   ├── 03_cleaning_excess.ipynb       # Calculate excess returns
│   └── 04_clean_1year.ipynb           # Final 1-year data preparation
│
├── src/                      # Source code
│   ├── models.py             # All ML models
│   ├── config.py             # Configuration
│   └── utils.py              # Helper functions
│
├── scripts/                  # Executable scripts
│   └── train_models.py       # Train all models
│
├── results/                  # Model outputs
│   ├── figures/              # Plots and visualizations
│   ├── tables/               # Performance metrics
│   └── models/               # Saved model files
│
└── paper/                    # LaTeX paper
    ├── main.tex
    ├── main.pdf
    └── Tables/
```

---

## Model Performance

| Model | 1-Day Accuracy | 1-Day ROC-AUC | 1-Year Accuracy | 1-Year ROC-AUC |
|-------|---------------|---------------|-----------------|----------------|
| **Gradient Boosting** | **70.26%** | 0.654 | 63.56% | 0.674 |
| **Random Forest** | 66.86% | 0.659 | **65.37%** | **0.691** |
| Elastic-Net Logistic | 59.14% | 0.604 | 61.97% | 0.630 |
| Decision Tree | 58.80% | 0.590 | 61.52% | 0.643 |

---

## Methodology

### Data Sources
1. **Bloomberg IPO Data (1998-2020):** Firm characteristics and financial metrics
2. **IPOscoop.com:** First-day returns data  
3. **Yahoo Finance:** One-year post-IPO stock prices and S&P 500 returns

### Models
1. **Elastic-Net Logistic Regression** - Regularized linear baseline
2. **Decision Tree** - Interpretable non-linear model
3. **Random Forest** - Bagging ensemble method
4. **Gradient Boosting** - Sequential boosting ensemble

### Training
- **Cross-validation:** 10-fold stratified
- **Class imbalance:** SMOTE for 1-day (75.1% imbalance), balanced weights for 1-year
- **Hyperparameter optimization:** Grid search with nested CV
- **Evaluation:** Accuracy, ROC-AUC, F1-score, Precision, Recall

---

## Citation
```bibtex
@article{garcia2025ipo,
  title={How does the importance of pre-IPO attributes change across different investment horizons},
  author={Garcia, Daniel and Pearce, Charles and Stomps, Lars and Hu, Johnny},
  year={2025},
  institution={Erasmus University Rotterdam}
}
```

---

## License

This project is licensed under the MIT License.

---

## Contact

- **Daniel Garcia:** 637987db@eur.nl
- **Charles Pearce:** 695309cp@eur.nl
- **Lars Stomps:** 678137ls@eur.nl
- **Johnny Hu:** 677926jh@eur.nl
