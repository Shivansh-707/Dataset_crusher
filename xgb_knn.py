import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from xgboost import XGBClassifier
import optuna
import warnings
warnings.filterwarnings('ignore')

# Fill these out bro
target = ""
evaluation_metric = ""

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Separate ID and target
train_ids = train['id'] if 'id' in train.columns else train.index
test_ids = test['id'] if 'id' in test.columns else test.index

y = train[target]
X = train.drop(columns=[target, 'id'] if 'id' in train.columns else [target])
X_test = test.drop(columns=['id'] if 'id' in test.columns else [])

# Label encoding
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

# Optuna objective
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'random_state': 42,
        'eval_metric': 'logloss',
        'verbosity': 0
    }
    
    cv_scores = []
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)
        
        imputer = KNNImputer(n_neighbors=5)
        X_tr_imputed = imputer.fit_transform(X_tr_scaled)
        X_val_imputed = imputer.transform(X_val_scaled)
        
        model = XGBClassifier(**params)
        model.fit(X_tr_imputed, y_tr)
        
        if evaluation_metric == 'accuracy':
            score = model.score(X_val_imputed, y_val)
        else:
            preds = model.predict_proba(X_val_imputed)[:, 1]
            from sklearn.metrics import roc_auc_score
            score = roc_auc_score(y_val, preds)
        
        cv_scores.append(score)
    
    return np.mean(cv_scores)

# Tune hyperparameters
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

best_params = study.best_params
best_params['random_state'] = 42
best_params['eval_metric'] = 'logloss'
best_params['verbosity'] = 0

# Train with best params and get OOF predictions
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    imputer = KNNImputer(n_neighbors=5)
    X_tr_imputed = imputer.fit_transform(X_tr_scaled)
    X_val_imputed = imputer.transform(X_val_scaled)
    X_test_imputed = imputer.transform(X_test_scaled)
    
    model = XGBClassifier(**best_params)
    model.fit(X_tr_imputed, y_tr)
    
    oof_preds[val_idx] = model.predict_proba(X_val_imputed)[:, 1]
    test_preds += model.predict_proba(X_test_imputed)[:, 1] / 3

# Save OOF predictions
try:
    oof_df = pd.read_csv('all_oof_preds.csv')
    oof_df['xgb_oof'] = oof_preds
except FileNotFoundError:
    oof_df = pd.DataFrame({'xgb_oof': oof_preds})

oof_df.to_csv('all_oof_preds.csv', index=False)

# Save test predictions
try:
    test_df = pd.read_csv('all_test_preds.csv')
    test_df['xgb_test'] = test_preds
except FileNotFoundError:
    test_df = pd.DataFrame({'xgb_test': test_preds})

test_df.to_csv('all_test_preds.csv', index=False)
