import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
import optuna
from gplearn.genetic import SymbolicTransformer
import warnings
warnings.filterwarnings('ignore')

# Fill these out bro 
target = "diagnosed_diabetes"
evaluation_metric = "roc_auc"

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


print("lets check for NaN values...")
if X.isnull().any().any() or X_test.isnull().any().any():
    print("Found NaN values, filling with median")
    X = X.fillna(X.median())
    X_test = X_test.fillna(X.median())


print("starting the god of FE , that is GP")
# Genetic Programming Feature Engineering
gp = SymbolicTransformer(
    generations=20,
    population_size=2000,
    tournament_size=20,
    stopping_criteria=0.01,
    function_set=['add', 'sub', 'mul', 'div'],
    parsimony_coefficient=0.001,
    max_samples=0.9,
    verbose=0,
    random_state=42,
    n_jobs=-1
)

gp_features = gp.fit_transform(X, y)
gp_features_test = gp.transform(X_test)

X_gp = np.hstack([X.values, gp_features])
X_test_gp = np.hstack([X_test.values, gp_features_test])

# Optuna objective
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'random_state': 42,
        'verbose': -1
    }
    
    cv_scores = []
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    for train_idx, val_idx in skf.split(X_gp, y):
        X_tr, X_val = X_gp[train_idx], X_gp[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = LGBMClassifier(**params)
        model.fit(X_tr, y_tr)
        
        if evaluation_metric == 'accuracy':
            score = model.score(X_val, y_val)
        else:
            preds = model.predict_proba(X_val)[:, 1]
            from sklearn.metrics import roc_auc_score
            score = roc_auc_score(y_val, preds)
        
        cv_scores.append(score)
    
    return np.mean(cv_scores)

# Tune hyperparameters
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

best_params = study.best_params
best_params['random_state'] = 42
best_params['verbose'] = -1

# Train with best params and get OOF predictions
oof_preds = np.zeros(len(X_gp))
test_preds = np.zeros(len(X_test_gp))

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_gp, y)):
    X_tr, X_val = X_gp[train_idx], X_gp[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = LGBMClassifier(**best_params)
    model.fit(X_tr, y_tr)
    
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_test_gp)[:, 1] / 3 #gotta divide the folds 

# Save OOF predictions
try:
    oof_df = pd.read_csv('all_oof_preds.csv')
    oof_df['lgb_oof'] = oof_preds
except FileNotFoundError:
    oof_df = pd.DataFrame({'lgb_oof': oof_preds})

oof_df.to_csv('all_oof_preds.csv', index=False)

# Save test predictions
try:
    test_df = pd.read_csv('all_test_preds.csv')
    test_df['lgb_test'] = test_preds
except FileNotFoundError:
    test_df = pd.DataFrame({'lgb_test': test_preds})

test_df.to_csv('all_test_preds.csv', index=False)