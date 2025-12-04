# MetaBoost 🚀
**A Dataset Crusher — Insert raw data → Get a great submission file**

---

## 📖 Overview
MetaBoost is a production-ready, advanced ensemble learning pipeline for **binary classification** on tabular data.  
It combines multiple gradient-boosting frameworks with meta-learners using multi-layer stacking and blending to maximize predictive performance.

Although optimized for medical prediction tasks (e.g., diabetes detection), the modular design generalizes to **any binary classification problem** with structured data.

---

## 🎯 Key Highlights
- **Diverse ensemble models**: LightGBM, CatBoost, XGBoost with different preprocessing strategies  
- **Meta-learning architecture**: Dual-layer stacking with LightGBM and Ridge Regression as meta-learners  
- **Hyperparameter optimization**: Automated with Optuna for each major component  
- **Genetic programming features**: Symbolic feature engineering using evolutionary methods  
- **Automatic weighting**: Cross-validation–based weighting for final prediction blending

---

## 🧠 Architecture

### Layer 1 — Base Learners
Each base learner generates **out-of-fold (OOF)** predictions on the train set and test predictions. OOF predictions become features for the meta-learning layer.

| Base Model | File | Description | Key Techniques | Optuna Trials |
|------------|------|-------------|----------------|---------------|
| LightGBM + Genetic Programming | `lgb_gp.py` | Hybrid feature engineering + boosting | Symbolic feature synthesis via genetic programming (`SymbolicTransformer`); GP-derived features concatenated with original features | 50 |
| CatBoost Native | `catboost_raw.py` | Native handling of categorical variables | Uses CatBoost’s built-in categorical processing (no manual encoding) | 25 |
| XGBoost + KNN Imputation | `xgb_knn.py` | Gradient boosting with robust missing-data strategy | StandardScaler + KNN imputation (k=5); XGBoost-specific tuning (gamma, min_child_weight, etc.) | 50 |

### Layer 2 — Meta Learners
The meta_learner consumes OOF predictions from base models and learns how to optimally combine them.

| Meta Model | File | Description | Purpose | Optuna Trials |
|------------|------|-------------|--------|---------------|
| LightGBM Meta-Learner | `meta_learner.py` | Shallow LightGBM on top of base predictions | Captures non-linear interactions between base outputs while controlling overfitting | 30 |
| Ridge Regression | `meta_learner.py` | Linear model with L2 regularization | Stable, interpretable linear blending; can be clipped for probability calibration | – (configurable) |

---

## 🧩 Ensemble Blending
Final predictions are produced via a weighted average of meta-learners (or base models, depending on configuration). Weights are proportional to model performance measured on cross-validation.

**Formula:**
Where:  
- `Pi` = predictions from model i (base or meta)  
- `wi` = weight derived from CV performance of model i  
- `sum(wi) = 1`

---

## 🏗️ Pipeline Workflow
1. Load raw tabular data (`train.csv`, `test.csv`)  
2. Apply preprocessing & feature engineering:
   - Standard scaling  
   - KNN imputation (for XGBoost branch)  
   - Genetic programming–based symbolic features (for LightGBM GP branch)  
3. Train base learners with Optuna-tuned hyperparameters  
4. Generate OOF predictions for stacking  
5. Train meta-learners on OOF predictions  
6. Compute performance-based weights  
7. Produce final submission file via weighted blending


