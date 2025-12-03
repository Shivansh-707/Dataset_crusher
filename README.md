# MetaBoost , a dataset crusher 
Insert raw data --> get a great submission file 


Advanced Ensemble Learning for Binary Classification
A production-ready machine learning pipeline featuring a sophisticated stacking/blending ensemble that combines LightGBM with genetic programming, CatBoost native categorical handling, XGBoost with KNN imputation, and dual meta-learners (LightGBM + Ridge regression) for robust binary classification on tabular data.

🎯 Project Overview
This repository implements an advanced ensemble architecture that achieves superior predictive performance through diversity across multiple boosting frameworks and preprocessing strategies. The pipeline is particularly effective for medical prediction tasks (e.g., diabetes diagnosis) but generalizes to any binary classification problem with tabular data.

Key Innovation: Multi-layer stacking with heterogeneous base learners optimized via Optuna hyperparameter tuning, followed by weighted meta-learner blending based on cross-validation performance.

🏗️ Architecture
Base Learner Models (Layer 1)
Each base learner generates out-of-fold (OOF) predictions on training data and test predictions, which serve as input to the meta-learning layer:

**LightGBM + Genetic Programming (`lgb_gp.py`)**


Symbolic feature engineering via genetic programming (SymbolicTransformer)
Generates synthetic features through evolutionary algorithms
Base features concatenated with GP-derived features for improved signal
50 Optuna trials for hyperparameter optimization


**CatBoost Native (catboost_raw.py)**


Built-in categorical feature handling (no manual encoding)
Avoids categorical encoding bias introduced by label encoding
25 Optuna trials optimizing tree depth, learning rate, and regularization
Efficient handling of mixed data types



**XGBoost + KNN Imputation (xgb_knn.py)**


StandardScaler normalization + KNN imputation (k=5 neighbors)
Leverages manifold structure for missing value handling
50 Optuna trials with XGBoost-specific parameters (gamma, min_child_weight)
Complementary to gradient-based feature scaling in other models



**Meta-Learner (Layer 2)**
meta_learner.py trains two secondary models on base learner predictions:

**LightGBM Meta-Learner**

30 Optuna trials with shallow hyperparameters (max_depth=2-5)
Shallow trees reduce overfitting on OOF predictions
Learns feature interactions between base models

**Ridge Regression Meta-Learner**

Linear blending with L2 regularization
Output clipped to [0, 1] for probability calibration (Though, this depends on the evaluation metric )
purpose to include a linear model as meta learner : Provides stability and interpretability


**Ensemble Weighting: Final predictions are weighted by cross-validation performance**

Ensemble submission file = weighted average of both meta learners based on their CV score ( eg 0.55 * LGB + 0.45 * ridge ) 

 
