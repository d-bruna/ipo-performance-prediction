# Predicting IPO Performance Across Investment Horizons

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)


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
ipo-performance-prediction/
├── data/
│   └── processed/
│       └── ipo_data_with_1year_excess_returns_CLEANED_FINAL.csv
│
├── notebooks/
│   ├── 01_data_scrape.ipynb              # Scrape Yahoo Finance data
│   ├── 02_data_cleaning.ipynb            # Initial data cleaning
│   ├── 03_cleaning_excess.ipynb          # Calculate excess returns
│   └── 04_clean_1year.ipynb              # Final 1-year data preparation
│
├── src/
│   ├── config.py                         # Configuration and hyperparameters
│   └── models.py                         # ML models and evaluation functions
│
├── scripts/
    └── train_models.py                   # Main training script
```


## Model Performance

| Model | 1-Day Accuracy | 1-Day ROC-AUC | 1-Year Accuracy | 1-Year ROC-AUC |
|-------|---------------|---------------|-----------------|----------------|
| **Gradient Boosting** | **70.26%** | 0.654 | 63.56% | 0.674 |
| **Random Forest** | 66.86% | 0.659 | **65.37%** | **0.691** |
| Elastic-Net Logistic | 59.14% | 0.604 | 61.97% | 0.630 |
| Decision Tree | 58.80% | 0.590 | 61.52% | 0.643 |


### Confusion Matrices

#### 1-Day Predictions

**Decision Tree:**
| | Predicted Underperform | Predicted Outperform |
|---|---|---|
| **Actual Underperform (219)** | 79 | 140 |
| **Actual Outperform (662)** | 119 | 543 |

- Accuracy: 70.6% | Precision (Outperform): 0.79 | Recall (Outperform): 0.82

**Random Forest:**
| | Predicted Underperform | Predicted Outperform |
|---|---|---|
| **Actual Underperform (219)** | 74 | 145 |
| **Actual Outperform (662)** | 147 | 515 |

- Accuracy: 66.9% | Precision (Outperform): 0.78 | Recall (Outperform): 0.78

**Gradient Boosting:**
| | Predicted Underperform | Predicted Outperform |
|---|---|---|
| **Actual Underperform (219)** | 79 | 140 |
| **Actual Outperform (662)** | 119 | 543 |

- Accuracy: 70.3% | TN: 79, FP: 140, FN: 119, TP: 543
- Precision (Outperform): 0.79 | Recall (Outperform): 0.82 | F1 (Outperform): 0.80
- Precision (Underperform): 0.39 | Recall (Underperform): 0.36 | F1 (Underperform): 0.38

**Elastic-Net Logistic:**
| | Predicted Underperform | Predicted Outperform |
|---|---|---|
| **Actual Underperform (219)** | 45 | 174 |
| **Actual Outperform (662)** | 186 | 476 |

- Accuracy: 59.1% | Precision (Outperform): 0.73 | Recall (Outperform): 0.72

---

#### 1-Year Predictions

**Decision Tree:**
| | Predicted Underperform | Predicted Outperform |
|---|---|---|
| **Actual Underperform (496)** | 312 | 184 |
| **Actual Outperform (385)** | 155 | 230 |

- Accuracy: 61.5% | Precision (Underperform): 0.67 | Recall (Underperform): 0.63

**Random Forest:**
| | Predicted Underperform | Predicted Outperform |
|---|---|---|
| **Actual Underperform (496)** | 341 | 155 |
| **Actual Outperform (385)** | 150 | 235 |

- Accuracy: 65.4% | TN: 341, FP: 155, FN: 150, TP: 235
- Precision (Underperform): 0.69 | Recall (Underperform): 0.69 | F1 (Underperform): 0.69
- Precision (Outperform): 0.60 | Recall (Outperform): 0.61 | F1 (Outperform): 0.60

**Gradient Boosting:**
| | Predicted Underperform | Predicted Outperform |
|---|---|---|
| **Actual Underperform (496)** | 347 | 149 |
| **Actual Outperform (385)** | 172 | 213 |

- Accuracy: 63.6% | TN: 347, FP: 149, FN: 172, TP: 213
- Precision (Underperform): 0.67 | Recall (Underperform): 0.70 | F1 (Underperform): 0.68
- Precision (Outperform): 0.59 | Recall (Outperform): 0.56 | F1 (Outperform): 0.57

**Elastic-Net Logistic:**
| | Predicted Underperform | Predicted Outperform |
|---|---|---|
| **Actual Underperform (496)** | 321 | 175 |
| **Actual Outperform (385)** | 160 | 225 |

- Accuracy: 62.0% | Precision (Underperform): 0.67 | Recall (Underperform): 0.65

---


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

## Contact

- **Daniel Garcia:** 637987db@eur.nl
- **Charles Pearce:** 695309cp@eur.nl
- **Lars Stomps:** 678137ls@eur.nl
- **Johnny Hu:** 677926jh@eur.nl
