MetaBoost 🚀
"A Dataset Crusher — Insert raw data → Get a great submission file"

📖 Overview
MetaBoost is a production-ready advanced ensemble learning pipeline for binary classification tasks on tabular data. It combines the strengths of multiple gradient boosting frameworks and meta-learners through multi-layer stacking and blending to maximize predictive performance.

While optimized for medical prediction tasks (for example, diabetes detection), its modular design generalizes to any binary classification problem with structured data.

🎯 Key Highlights
Diverse Ensemble Models: LightGBM, CatBoost, XGBoost with different preprocessing strategies.

Meta-Learning Architecture: Dual-layer stacking with LightGBM and Ridge Regression as meta-learners.

Hyperparameter Optimization: Automated with Optuna on every major component.

Genetic Programming Features: Symbolic feature engineering using evolutionary methods.

Automatic Weighting: Cross-validation–based weighting for final prediction blending.

🧠 Architecture
Layer 1: Base Learners
Each base learner generates out-of-fold (OOF) predictions on the training data and predictions on the test data. These predictions become features for the meta-learning layer.

Base Model	File	Description	Key Techniques	Optuna Trials
LightGBM + Genetic Programming	lgb_gp.py	Hybrid feature engineering + boosting	Symbolic feature synthesis via genetic programming (SymbolicTransformer); GP-derived synthetic features concatenated with original features	50
CatBoost Native	catboost_raw.py	Native handling of categorical variables	Uses CatBoost’s built-in categorical processing (no manual encoding), avoids label-encoding bias, handles mixed data types	25
XGBoost + KNN Imputation	xgb_knn.py	Gradient boosting with robust missing-data strategy	StandardScaler + KNN imputation (k = 5); exploits local manifold for missing value handling; XGBoost-specific tuning (gamma, min_child_weight, etc.)	50
Layer 2: Meta Learners
The meta_learner consumes OOF predictions from the base models and learns how to optimally combine them.

Meta Model	File	Description	Purpose	Optuna Trials
LightGBM Meta-Learner	meta_learner.py	Shallow LightGBM on top of base model predictions	Captures non-linear interactions between base model outputs while controlling overfitting with small depth trees	30
Ridge Regression	meta_learner.py	Linear model with L2 regularization	Adds a stable, interpretable linear blending component; output can be clipped to if needed for probability calibration	– (configurable)
Ensemble Blending
Final predictions are produced via a weighted average of the meta-learners (or directly of the base models, depending on your configuration), where each weight is proportional to model performance on cross-validation.

Plain-text formula (so GitHub renders it safely):

Final_Prediction = w1 * P1 + w2 * P2 + ... + wn * Pn

where:

Pi = predictions from model i (base or meta),

wi = weight derived from cross-validation performance of model i,

and sum(wi) = 1.

