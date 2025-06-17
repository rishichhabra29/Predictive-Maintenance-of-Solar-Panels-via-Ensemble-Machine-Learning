import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

import optuna
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor

from feature_engineering import feature_engineering

def tune_xgb(X, y, n_trials=20):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
        }
        model = XGBRegressor(**params, random_state=42, n_jobs=-1)
        rmses = []
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        for ti, vi in kf.split(X):
            model.fit(X[ti], y[ti])
            preds = model.predict(X[vi])
            rmses.append(np.sqrt(mean_squared_error(y[vi], preds)))
        return np.mean(rmses)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def tune_lgb(X, y, n_trials=20):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
        }
        model = lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1)
        rmses = []
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        for ti, vi in kf.split(X):
            model.fit(X[ti], y[ti])
            preds = model.predict(X[vi])
            rmses.append(np.sqrt(mean_squared_error(y[vi], preds)))
        return np.mean(rmses)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def target_encode(train, test, col, target, n_splits=5):
    oof = pd.Series(index=train.index, dtype=float)
    kf  = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for ti, vi in kf.split(train):
        means = train.iloc[ti].groupby(col)[target].mean()
        oof.iloc[vi] = train.iloc[vi][col].map(means)
    train[f'{col}_te']    = oof
    global_means          = train.groupby(col)[target].mean()
    test[f'{col}_te']     = test[col].map(global_means)
    train.drop(columns=[col], inplace=True)
    test.drop(columns=[col], inplace=True)

def main():
    # 1) Load data
    train = pd.read_csv('train.csv')
    test  = pd.read_csv('test.csv')
    ids   = test['id']

    # 2) Target-encode string_id, then FE
    target_encode(train, test, 'string_id', 'efficiency')
    train = feature_engineering(train)
    test  = feature_engineering(test)

    # 3) Prepare DataFrames
    y         = train['efficiency'].values
    X_df      = train.drop(['id','efficiency'], axis=1)
    X_test_df = test.drop(['id'], axis=1)

    # 4) Skew correction & PowerTransform
    skew = ['irradiance','soiling_ratio','wind_speed','pressure']
    pt   = PowerTransformer(method='yeo-johnson')
    for col in skew:
        tr  = pd.to_numeric(X_df[col], errors='coerce'); med = tr.median()
        X_df[col]      = tr.fillna(med)
        te  = pd.to_numeric(X_test_df[col], errors='coerce')
        X_test_df[col] = te.fillna(med)
    X_df[skew]      = pt.fit_transform(X_df[skew])
    X_test_df[skew] = pt.transform(X_test_df[skew])

    # 5) Drop non-numeric & align columns
    X_df      = X_df.select_dtypes(include='number')
    X_test_df = X_test_df[X_df.columns]

    # 6) Convert to NumPy and scale
    X      = StandardScaler().fit_transform(X_df.values)
    X_test = StandardScaler().fit_transform(X_test_df.values)

    # 7) Hyperparameter tuning on 20% subsample
    idx        = np.random.choice(len(X), size=int(0.2*len(X)), replace=False)
    xgb_params = tune_xgb(X[idx], y[idx], n_trials=10)
    lgb_params = tune_lgb(X[idx], y[idx], n_trials=10)

    # 8) Instantiate tuned base learners
    xgb = XGBRegressor(**xgb_params, random_state=42, n_jobs=-1)
    lgbm= lgb.LGBMRegressor(**lgb_params, random_state=42, n_jobs=-1)
    cat = CatBoostRegressor(iterations=500, learning_rate=0.05,
                            depth=6, random_seed=42, verbose=False)

    # 9) Out-of-Fold stacking
    kf          = KFold(n_splits=5, shuffle=True, random_state=42)
    base_models = [xgb, lgbm, cat]
    oof_preds   = np.zeros((len(X), len(base_models)))
    test_preds  = np.zeros((len(X_test), len(base_models)))

    for i, mdl in enumerate(base_models):
        fold_preds = np.zeros((len(X_test), kf.n_splits))
        for fold, (ti, vi) in enumerate(kf.split(X)):
            X_tr, X_val = X[ti], X[vi]
            y_tr, y_val = y[ti], y[vi]

            if isinstance(mdl, lgb.LGBMRegressor):
                mdl.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=30),
                               lgb.log_evaluation(period=0)]
                )
            elif isinstance(mdl, CatBoostRegressor):
                mdl.fit(
                    X_tr, y_tr,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=30,
                    verbose=False
                )
            else:
                mdl.fit(X_tr, y_tr)

            oof_preds[vi, i]    = mdl.predict(X_val)
            fold_preds[:, fold] = mdl.predict(X_test)

        test_preds[:, i] = fold_preds.mean(axis=1)

    # 10) Meta-learner
    meta      = Ridge(alpha=1.0)
    meta.fit(oof_preds, y)
    oof_out   = meta.predict(oof_preds)
    rmse_oof  = np.sqrt(mean_squared_error(y, oof_out))
    score_oof = 100 * (1 - rmse_oof)
    print(f"Stacking OOF RMSE: {rmse_oof:.5f}, Score: {score_oof:.2f}")

    # 11) Blend-weight optimization
    base_avg   = oof_preds.mean(axis=1)
    best_alpha, best_rmse = 0.5, rmse_oof
    for α in np.linspace(0,1,11):
        blend = α * oof_out + (1-α) * base_avg
        rm = np.sqrt(mean_squared_error(y, blend))
        if rm < best_rmse:
            best_rmse, best_alpha = rm, α
    print(f"Best blend α={best_alpha:.2f}, RMSE={best_rmse:.5f}, Score={100*(1-best_rmse):.2f}")

    # 12) Final predict & save
    final_preds = best_alpha * meta.predict(test_preds) + (1-best_alpha) * test_preds.mean(axis=1)
    pd.DataFrame({'id': ids, 'efficiency': final_preds}) \
      .to_csv('submission.csv', index=False)

    print("submission.csv generated successfully.")

if __name__ == '__main__':
    main()
