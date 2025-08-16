# Lab05_Aromal_Gigi# Lab 5 — Company Bankruptcy Prediction

## Overview

The pipeline covers:
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Feature selection
- Hyperparameter tuning
- Model training
- Evaluation
- Interpretability (SHAP)
- Population Stability Index (PSI)
- Report generation

---

## Dataset
- **Source:** Provided as `bankruptcy.csv`
- **Target Column:** `Bankrupt?`  
- **Class Imbalance:** ~3.23% positive class

---

## Models Trained
- Logistic Regression
- Random Forest
- XGBoost

---

## Key Decisions
1. **Imbalance Handling**
   - Used `class_weight="balanced"` (LR, RF)
   - Used `scale_pos_weight` for XGB
   - No SMOTE (to avoid synthetic noise)
2. **Feature Selection**
   - Removed highly correlated features (|r| > 0.90)
   - VIF pruning for Logistic Regression
3. **Primary Metric**
   - **PR-AUC** (Precision-Recall Area Under Curve) chosen due to high imbalance
4. **Tuning**
   - RandomizedSearchCV with Stratified K-Fold
   - PR-AUC used for scoring

---

## Results

### Metrics Table
| Model              | ROC AUC (Test) | PR AUC (Test) | Brier (Test) | F1 (Test) | F2 (Test) | Recall@P80 |
|--------------------|---------------:|--------------:|-------------:|----------:|----------:|-----------:|
| RandomForest       | **0.9520**     | **0.5173**    | **0.0234**   | 0.4400    | 0.4151    | 0.1818     |
| XGBoost            | 0.9559         | 0.5073        | 0.0320       | **0.5067**| **0.6032**| **0.2000** |
| LogisticRegression | 0.9454         | 0.4214        | 0.0897       | 0.3072    | 0.4989    | 0.1273     |

**Best Model:** **Random Forest** (highest PR-AUC and balanced generalization)

---

## Visualizations
- **Class Balance** → Showed heavy imbalance (~3% positive)
- **Correlation Heatmap** → Identified redundant features for removal
- **ROC Curves** → RandomForest & XGBoost showed high separability
- **PR Curves** → RandomForest had better precision-recall balance
- **Calibration Curves** → Both RandomForest & XGBoost well-calibrated
- **SHAP Plots** → Top features influencing predictions explained
- **PSI** → Validated train-test stability

---

## Recommended Model for Deployment
**Random Forest**  
- Strong PR-AUC (minority class focus)  
- Good calibration  
- Robust to feature correlation  
- Interpretable with SHAP

---

## How to Run
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Run pipeline
python training_pipeline.py 

```
