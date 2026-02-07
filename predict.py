import os
os.environ['SCIPY_ARRAY_API'] = '1'

import json
import numpy as np
import pandas as pd
import joblib
import sys

MODEL_DIR = 'models'


def load_model():
    pipe = joblib.load(f'{MODEL_DIR}/best_pipeline.pkl')
    with open(f'{MODEL_DIR}/metadata.json') as f:
        meta = json.load(f)
    return pipe, meta


def predict(df, pipe, threshold):
    scores = pipe.predict_proba(df)[:, 1]
    preds = (scores >= threshold).astype(int)
    return preds, scores


def main():
    if len(sys.argv) < 2:
        print('usage: python predict.py <csv_file>')
        print('  csv should have the same columns as creditcard.csv (minus Class)')
        sys.exit(1)

    filepath = sys.argv[1]
    df = pd.read_csv(filepath)

    # drop Class column if it's there
    if 'Class' in df.columns:
        df = df.drop('Class', axis=1)

    pipe, meta = load_model()
    threshold = meta['threshold']
    expected_cols = meta.get('features', [])

    if expected_cols:
        missing = set(expected_cols) - set(df.columns)
        if missing:
            print(f'error: missing columns: {missing}')
            sys.exit(1)
        df = df[expected_cols]

    preds, scores = predict(df, pipe, threshold)

    df['fraud_score'] = scores
    df['prediction'] = preds
    df['prediction'] = df['prediction'].map({0: 'legit', 1: 'fraud'})

    n_fraud = (preds == 1).sum()
    print(f'{len(df)} transactions, {n_fraud} flagged as fraud')
    print(f'threshold: {threshold}')

    if n_fraud > 0:
        print(f'\nflagged transactions:')
        print(df[df['prediction'] == 'fraud'][['fraud_score']].to_string())

    output = filepath.replace('.csv', '_predictions.csv')
    df.to_csv(output, index=False)
    print(f'\nsaved to {output}')


if __name__ == '__main__':
    main()
