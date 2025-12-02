import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier, Pool
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

# Detect categorical features
cat_features = [col for col in X.columns if X[col].dtype == 'object']
cat_indices = [X.columns.get_loc(col) for col in cat_features]

# Optuna objective
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 150, 600),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 4, 8),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 128),
        'random_strength': trial.suggest_float('random_strength', 0, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'random_seed': 42,
        'verbose': 0,
        'thread_count': 6

    }
    
    cv_scores = []
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        train_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
        val_pool = Pool(X_val, y_val, cat_features=cat_indices)
        
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool, verbose=0)
        
        if evaluation_metric == 'accuracy':
            preds = model.predict(val_pool)
            score = (preds == y_val).mean()
        else:
            preds = model.predict_proba(val_pool)[:, 1]
            from sklearn.metrics import roc_auc_score
            score = roc_auc_score(y_val, preds)
        
        cv_scores.append(score)
    
    return np.mean(cv_scores)

# Tune hyperparameters
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=25, show_progress_bar=True)

best_params = study.best_params
best_params['random_seed'] = 42
best_params['verbose'] = 0

# Train with best params and get OOF predictions
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    train_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
    val_pool = Pool(X_val, cat_features=cat_indices)
    test_pool = Pool(X_test, cat_features=cat_indices)
    
    model = CatBoostClassifier(**best_params)
    model.fit(train_pool, verbose=0)
    
    oof_preds[val_idx] = model.predict_proba(val_pool)[:, 1]
    test_preds += model.predict_proba(test_pool)[:, 1] / 3

# Save OOF predictions
try:
    oof_df = pd.read_csv('all_oof_preds.csv')
    oof_df['cat_oof'] = oof_preds
except FileNotFoundError:
    oof_df = pd.DataFrame({'cat_oof': oof_preds})

oof_df.to_csv('all_oof_preds.csv', index=False)

# Save test predictions
try:
    test_df = pd.read_csv('all_test_preds.csv')
    test_df['cat_test'] = test_preds
except FileNotFoundError:
    test_df = pd.DataFrame({'cat_test': test_preds})

test_df.to_csv('all_test_preds.csv', index=False)
