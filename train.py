import os
os.environ['SCIPY_ARRAY_API'] = '1'
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from evaluate import (
    plot_confusion_matrix, plot_pr_curves,
    plot_roc_curves, plot_feature_importance, print_report
)

warnings.filterwarnings('ignore')

DATA_PATH = 'data/creditcard.csv'
FIG_DIR = 'figures'
SEED = 42


def load_data():
    df = pd.read_csv(DATA_PATH)

    # scale the only two raw features, V1-V28 are already PCA'd
    scaler = RobustScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])
    df['Time'] = scaler.fit_transform(df[['Time']])

    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    print(f'train: {len(X_train)} ({y_train.mean():.4%} fraud)')
    print(f'test:  {len(X_test)} ({y_test.mean():.4%} fraud)')
    return X_train, X_test, y_train, y_test, X.columns.tolist()


def get_models():
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=SEED, class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=SEED, class_weight='balanced',
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=100,
            random_state=SEED, eval_metric='aucpr',
            use_label_encoder=False, n_jobs=-1
        ),
    }


def train_with_smote(model, X_train, y_train):
    pipe = ImbPipeline([
        ('smote', SMOTE(random_state=SEED)),
        ('model', model)
    ])
    pipe.fit(X_train, y_train)
    return pipe


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    print('loading data...\n')
    X_train, X_test, y_train, y_test, feat_names = load_data()

    models = get_models()
    all_scores = {}
    best_name, best_ap, best_pipe = None, 0, None

    for name, model in models.items():
        print(f'\ntraining {name}...')
        pipe = train_with_smote(model, X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_score = pipe.predict_proba(X_test)[:, 1]

        print_report(y_test, y_pred, name)
        all_scores[name] = y_score

        safe = name.lower().replace(' ', '_')
        plot_confusion_matrix(y_test, y_pred, title=name,
                              save_path=f'{FIG_DIR}/cm_{safe}.png')

        ap = average_precision_score(y_test, y_score)
        if ap > best_ap:
            best_ap = ap
            best_name = name
            best_pipe = pipe

    print('\ngenerating comparison plots...')
    plot_pr_curves(y_test, all_scores, save_path=f'{FIG_DIR}/precision_recall.png')
    plot_roc_curves(y_test, all_scores, save_path=f'{FIG_DIR}/roc_curves.png')

    inner_model = best_pipe.named_steps['model']
    plot_feature_importance(inner_model, feat_names,
                            save_path=f'{FIG_DIR}/feature_importance.png')

    print(f'\nbest model: {best_name} (AP={best_ap:.4f})')
    print(f'all figures in {FIG_DIR}/')


if __name__ == '__main__':
    main()
