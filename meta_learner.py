import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Fill these out bro 
target = ""
metric = ""

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
y = train[target]
test_ids = test.get('id', test.index)

oof_preds = pd.read_csv('all_oof_preds.csv')
test_preds = pd.read_csv('all_test_preds.csv')
X_meta, X_test = oof_preds.values, test_preds.values

# Optuna tuning functions
def tune_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 2, 5),
        'num_leaves': trial.suggest_int('num_leaves', 4, 16),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 0.5),
        'random_state': 42, 
        'verbose': -1
    }
    
    cv_scores = []
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    for tr_idx, val_idx in skf.split(X_meta, y):
        model = LGBMClassifier(**params)
        model.fit(X_meta[tr_idx], y.iloc[tr_idx])
        
        if metric == 'accuracy':
            preds = model.predict(X_meta[val_idx])
            score = accuracy_score(y.iloc[val_idx], preds)
        else:
            preds = model.predict_proba(X_meta[val_idx])[:, 1]
            score = roc_auc_score(y.iloc[val_idx], preds)
        
        cv_scores.append(score)
    
    return np.mean(cv_scores)

def tune_ridge(trial):
    params = {
        'alpha': trial.suggest_float('alpha', 0.01, 100.0, log=True), 
        'random_state': 42
    }
    
    cv_scores = []
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    for tr_idx, val_idx in skf.split(X_meta, y):
        model = Ridge(**params)
        model.fit(X_meta[tr_idx], y.iloc[tr_idx])
        preds = np.clip(model.predict(X_meta[val_idx]), 0, 1)
        
        if metric == 'accuracy':
            score = accuracy_score(y.iloc[val_idx], (preds > 0.5).astype(int))
        else:
            score = roc_auc_score(y.iloc[val_idx], preds)
        
        cv_scores.append(score)
    
    return np.mean(cv_scores)

# Run optuna tuning
print("Tuning LightGBM meta-learner...")
lgb_study = optuna.create_study(direction='maximize')
lgb_study.optimize(tune_lgb, n_trials=30, show_progress_bar=True)
lgb_params = {**lgb_study.best_params, 'random_state': 42, 'verbose': -1}
lgb_cv = lgb_study.best_value
print(f"Best LightGBM score: {lgb_cv:.6f}")

print("\nTuning Ridge meta-learner...")
ridge_study = optuna.create_study(direction='maximize')
ridge_study.optimize(tune_ridge, n_trials=30, show_progress_bar=True)
ridge_params = {**ridge_study.best_params, 'random_state': 42}
ridge_cv = ridge_study.best_value
print(f"Best Ridge score: {ridge_cv:.6f}")

# Get predictions from both meta-learners
lgb_oof = np.zeros(len(X_meta))
lgb_test = np.zeros(len(X_test))
ridge_oof = np.zeros(len(X_meta))
ridge_test = np.zeros(len(X_test))

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_meta, y)):
    # LightGBM
    lgb = LGBMClassifier(**lgb_params)
    lgb.fit(X_meta[tr_idx], y.iloc[tr_idx])
    lgb_oof[val_idx] = lgb.predict_proba(X_meta[val_idx])[:, 1]
    lgb_test += lgb.predict_proba(X_test)[:, 1] / 3
    
    # Ridge
    ridge = Ridge(**ridge_params)
    ridge.fit(X_meta[tr_idx], y.iloc[tr_idx])
    ridge_oof[val_idx] = np.clip(ridge.predict(X_meta[val_idx]), 0, 1)
    ridge_test += np.clip(ridge.predict(X_test), 0, 1) / 3

# Calculate weights based on CV performance
total = lgb_cv + ridge_cv
lgb_weight = lgb_cv / total
ridge_weight = ridge_cv / total

print(f"\nWeights: LightGBM={lgb_weight:.4f}, Ridge={ridge_weight:.4f}")

# Blend predictions
final_oof = lgb_weight * lgb_oof + ridge_weight * ridge_oof
final_test = lgb_weight * lgb_test + ridge_weight * ridge_test

if metric == 'accuracy':
    ensemble_score = accuracy_score(y, (final_oof > 0.5).astype(int))
else:
    ensemble_score = roc_auc_score(y, final_oof)

print(f"Final ensemble score: {ensemble_score:.6f}")

# Create submission
submission = pd.DataFrame({
    'id': test_ids,
    target: final_test
})
submission.to_csv('submission.csv', index=False)
print("\nSubmission saved")

# Get feature importance
final_lgb = LGBMClassifier(**lgb_params)
final_lgb.fit(X_meta, y)

importance_df = pd.DataFrame({
    'Model': oof_preds.columns,
    'Importance': final_lgb.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature importance:")
print(importance_df.to_string(index=False))

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Feature importance
ax1.barh(importance_df['Model'], importance_df['Importance'])
ax1.set_xlabel('Importance')
ax1.set_title('Feature Importance')
ax1.invert_yaxis()

# Model scores comparison
model_names = list(oof_preds.columns) + ['LGB Meta', 'Ridge Meta', 'Ensemble']
base_scores = []
for col in oof_preds.columns:
    if metric == 'accuracy':
        score = accuracy_score(y, (oof_preds[col] > 0.5).astype(int))
    else:
        score = roc_auc_score(y, oof_preds[col])
    base_scores.append(score)

all_scores = base_scores + [lgb_cv, ridge_cv, ensemble_score]
ax2.bar(model_names, all_scores)
ax2.set_ylabel(metric)
ax2.set_title('Model Scores')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('results.png', dpi=150)
print("\nPlot saved")
plt.show()
