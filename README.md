# Credit Card Fraud Detection
### Machine Learning Classification Project | Python · Scikit-learn · Pandas

---

## Overview

Credit card fraud costs billions of dollars globally every year. This project builds and evaluates machine learning models that automatically detect fraudulent transactions from historical data — without any manual review.

The core challenge is **extreme class imbalance**: only 0.17% of transactions are fraud. A naive model that labels everything as legitimate would be 99.83% accurate and completely useless. This project tackles that problem head-on.

---

## Dataset

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284,807 transactions from European cardholders
- 492 confirmed fraud cases (0.17%)
- Features V1–V28 are PCA-transformed to protect cardholder privacy
- Additional features: `Time`, `Amount`, `Class` (0 = legitimate, 1 = fraud)

---

## Project Structure

```
fraud-detection/
│
├── fraud_detection.ipynb   # Main notebook (full analysis)
├── README.md               # This file
└── creditcard.csv          # Dataset (download from Kaggle — not included)
```

---

## Methodology

| Step | What was done |
|---|---|
| EDA | Visualised class imbalance, amount distributions, feature correlations |
| Preprocessing | Scaled `Amount`, removed `Time`, stratified train/test split (80/20) |
| Imbalance handling | `class_weight='balanced'` to penalise missed fraud heavily |
| Model selection | Compared 3 classifiers using 5-fold stratified cross-validation |
| Evaluation | Confusion matrix, precision, recall, F1, ROC-AUC, Precision-Recall AUC |

---

## Results

### Cross-validation ROC-AUC (training set)

| Model | ROC-AUC | Std Dev |
|---|---|---|
| Logistic Regression | 0.9828 | ±0.0099 |
| Random Forest | 0.9760 | ±0.0141 |
| Decision Tree | 0.8775 | ±0.0264 |

### Test set performance (on unseen data)

| Model | Fraud caught | False alarms | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 90 / 98 | 1,441 | 0.06 | 0.92 | 0.9714 |
| Decision Tree | 83 / 98 | 1,091 | 0.07 | 0.85 | 0.9182 |
| **Random Forest** | **79 / 98** | **19** | **0.81** | **0.81** | **0.9798** |

### Winner: Random Forest

Logistic Regression had marginally higher cross-validation AUC (0.9828), but generated **1,441 false positives** on the test set — flagging innocent customers as fraudsters. Random Forest produced only **19 false alarms** while maintaining strong recall, making it the most practical model for real-world deployment.

> When a model says "fraud", you want it to be right. Random Forest was right 81% of the time. Logistic Regression was right only 6% of the time.

---

## Key Findings

- **V10, V4, and V14** were the most predictive features according to Random Forest feature importances
- **Accuracy is a misleading metric** for imbalanced problems — the Precision-Recall curve is more informative than ROC-AUC in this context
- `class_weight='balanced'` was more effective than under/over-sampling for this baseline
- Cross-validation scores alone can be misleading — always inspect the confusion matrix

---

## How to Run

1. Clone this repository
2. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project folder
3. Open `fraud_detection.ipynb` in Jupyter or Google Colab
4. Run all cells top to bottom

```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

---

## Libraries Used

- `pandas` / `numpy` — data manipulation
- `scikit-learn` — modelling, preprocessing, evaluation
- `matplotlib` / `seaborn` — visualisation
- `joblib` — model saving

---

## What I Learned

- How to handle severely imbalanced classification problems
- Why evaluation metrics must be chosen based on the business problem, not just model performance
- The difference between a model's ranking ability (ROC-AUC) and its real-world decision quality (confusion matrix)
- How ensemble methods like Random Forest reduce false positives compared to single classifiers

---

## Next Steps

- [ ] Try SMOTE (Synthetic Minority Oversampling) for explicit class balancing
- [ ] Add XGBoost / LightGBM for comparison
- [ ] Tune the decision threshold to optimise for recall vs precision
- [ ] Add SHAP values for model explainability

---

*Built as a portfolio project to demonstrate applied machine learning on a real-world imbalanced classification problem.*
