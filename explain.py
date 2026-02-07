import os
os.environ['SCIPY_ARRAY_API'] = '1'

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

DATA_PATH = 'data/creditcard.csv'
MODEL_DIR = 'models'
FIG_DIR = 'figures'
SEED = 42


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    print('loading model and data...')
    pipe = joblib.load(f'{MODEL_DIR}/best_pipeline.pkl')

    df = pd.read_csv(DATA_PATH)
    X = df.drop('Class', axis=1)
    y = df['Class']

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    scaler = pipe.named_steps['scaler']
    model = pipe.named_steps['model']
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns, index=X_test.index
    )

    sample_size = min(500, len(X_test_scaled))
    X_sample = X_test_scaled.sample(n=sample_size, random_state=SEED)

    print(f'computing SHAP values on {sample_size} samples...')

    # pick the right explainer based on model type
    from sklearn.linear_model import LogisticRegression
    if isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

    # beeswarm
    print('generating plots...')
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    # bar importance
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, plot_type='bar',
                      show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/shap_importance.png', dpi=150, bbox_inches='tight')
    plt.close()

    # waterfall for a specific fraud case
    fraud_mask = y_test.loc[X_sample.index] == 1
    if fraud_mask.any():
        fraud_idx = fraud_mask.idxmax()
        local_idx = X_sample.index.get_loc(fraud_idx)

        fig = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[local_idx],
                base_values=explainer.expected_value,
                data=X_sample.iloc[local_idx],
                feature_names=X_sample.columns.tolist()
            ),
            show=False, max_display=12
        )
        plt.tight_layout()
        plt.savefig(f'{FIG_DIR}/shap_waterfall_fraud.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f'done — SHAP plots in {FIG_DIR}/')


if __name__ == '__main__':
    main()
