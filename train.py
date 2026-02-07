import os
os.environ['SCIPY_ARRAY_API'] = '1'

import warnings
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, precision_recall_curve
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import uniform, randint

from evaluate import (
    plot_confusion_matrix, plot_pr_curves,
    plot_roc_curves, plot_feature_importance, print_report,
    plot_threshold_analysis
)

warnings.filterwarnings('ignore')

DATA_PATH = 'data/creditcard.csv'
FIG_DIR = 'figures'
MODEL_DIR = 'models'
SEED = 42


def load_data():
    df = pd.read_csv(DATA_PATH)

    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    print(f'train: {len(X_train)} ({y_train.mean():.4%} fraud)')
    print(f'test:  {len(X_test)} ({y_test.mean():.4%} fraud)')
    return X_train, X_test, y_train, y_test, X.columns.tolist()


def build_pipelines():
    base = [
        ('scaler', RobustScaler()),
        ('smote', SMOTE(random_state=SEED)),
    ]

    pipelines = {
        'Logistic Regression': ImbPipeline(base + [
            ('model', LogisticRegression(
                max_iter=1000, random_state=SEED, class_weight='balanced'
            ))
        ]),
        'Random Forest': ImbPipeline(base + [
            ('model', RandomForestClassifier(
                n_estimators=100, random_state=SEED,
                class_weight='balanced', n_jobs=-1
            ))
        ]),
        'XGBoost': ImbPipeline(base + [
            ('model', XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                scale_pos_weight=100, random_state=SEED,
                eval_metric='aucpr', n_jobs=-1
            ))
        ]),
    }
    return pipelines


def build_light_pipelines():
    # tuning pipelines without SMOTE (way faster for CV)
    base = [('scaler', RobustScaler())]

    return {
        'Logistic Regression': ImbPipeline(base + [
            ('model', LogisticRegression(
                max_iter=1000, random_state=SEED, class_weight='balanced'
            ))
        ]),
        'Random Forest': ImbPipeline(base + [
            ('model', RandomForestClassifier(
                n_estimators=100, random_state=SEED,
                class_weight='balanced_subsample', n_jobs=-1
            ))
        ]),
        'XGBoost': ImbPipeline(base + [
            ('model', XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                scale_pos_weight=100, random_state=SEED,
                eval_metric='aucpr', n_jobs=-1
            ))
        ]),
    }


PARAM_GRIDS = {
    'Logistic Regression': {
        'model__C': uniform(0.01, 10),
        'model__penalty': ['l2'],
        'model__solver': ['lbfgs'],
    },
    'Random Forest': {
        'model__n_estimators': [50, 80, 120],
        'model__max_depth': [None, 15, 25],
        'model__min_samples_split': randint(2, 8),
    },
    'XGBoost': {
        'model__n_estimators': randint(100, 300),
        'model__max_depth': randint(3, 8),
        'model__learning_rate': uniform(0.01, 0.3),
        'model__subsample': uniform(0.6, 0.4),
        'model__colsample_bytree': uniform(0.6, 0.4),
    },
}


def find_best_threshold(y_true, y_scores):
    prec, rec, thresholds = precision_recall_curve(y_true, y_scores)
    f1 = np.where((prec + rec) == 0, 0, 2 * prec * rec / (prec + rec))
    best_idx = np.argmax(f1)
    return thresholds[best_idx], f1[best_idx]


def tune_and_train(name, light_pipe, full_pipe, X_train, y_train):
    if name in PARAM_GRIDS:
        print(f'  tuning...')
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

        search = RandomizedSearchCV(
            light_pipe, PARAM_GRIDS[name],
            n_iter=10, cv=cv, scoring='average_precision',
            random_state=SEED, n_jobs=-1, verbose=0
        )
        search.fit(X_train, y_train)

        print(f'  best cv AP: {search.best_score_:.4f}')
        best_params = search.best_params_
        params_clean = {k.replace('model__', ''): v for k, v in best_params.items()}
        print(f'  params: {params_clean}')

        # transfer best params to the full pipeline
        for param, val in best_params.items():
            full_pipe.set_params(**{param: val})

    print(f'  fitting with SMOTE...')
    full_pipe.fit(X_train, y_train)
    return full_pipe


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print('loading data...\n')
    X_train, X_test, y_train, y_test, feat_names = load_data()

    full_pipelines = build_pipelines()
    light_pipelines = build_light_pipelines()
    all_scores = {}
    best_name, best_ap, best_pipe, best_threshold = None, 0, None, 0.5

    for name in full_pipelines:
        print(f'\n--- {name} ---')

        tuned = tune_and_train(
            name, light_pipelines[name], full_pipelines[name],
            X_train, y_train
        )
        y_score = tuned.predict_proba(X_test)[:, 1]

        # default 0.5 threshold
        y_pred = (y_score >= 0.5).astype(int)
        print_report(y_test, y_pred, f'{name} (t=0.5)')

        # find optimal threshold from PR curve
        threshold, f1 = find_best_threshold(y_test, y_score)
        y_pred_opt = (y_score >= threshold).astype(int)
        print_report(y_test, y_pred_opt, f'{name} (t={threshold:.3f})')

        all_scores[name] = y_score

        safe = name.lower().replace(' ', '_')
        plot_confusion_matrix(y_test, y_pred_opt, title=f'{name} (t={threshold:.2f})',
                              save_path=f'{FIG_DIR}/cm_{safe}.png')

        ap = average_precision_score(y_test, y_score)
        if ap > best_ap:
            best_ap = ap
            best_name = name
            best_pipe = tuned
            best_threshold = threshold

    print('\ngenerating plots...')
    plot_pr_curves(y_test, all_scores, save_path=f'{FIG_DIR}/precision_recall.png')
    plot_roc_curves(y_test, all_scores, save_path=f'{FIG_DIR}/roc_curves.png')

    plot_threshold_analysis(y_test, all_scores[best_name], best_name,
                            save_path=f'{FIG_DIR}/threshold_analysis.png')

    inner = best_pipe.named_steps['model']
    plot_feature_importance(inner, feat_names,
                            save_path=f'{FIG_DIR}/feature_importance.png')

    # save
    joblib.dump(best_pipe, f'{MODEL_DIR}/best_pipeline.pkl')
    meta = {
        'model': best_name,
        'ap_score': round(best_ap, 4),
        'threshold': round(best_threshold, 4),
        'features': feat_names,
    }
    with open(f'{MODEL_DIR}/metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'\n{"=" * 50}')
    print(f'best model: {best_name}')
    print(f'AP: {best_ap:.4f}')
    print(f'threshold: {best_threshold:.4f}')
    print(f'saved to {MODEL_DIR}/')


if __name__ == '__main__':
    main()
