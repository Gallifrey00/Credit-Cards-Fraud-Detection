# Credit Card Fraud Detection

Binary classification on the [Kaggle credit card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The dataset is extremely imbalanced (~0.17% fraud), so the whole pipeline is built around precision-recall rather than accuracy.

## Setup

```bash
pip install -r requirements.txt
```

Download `creditcard.csv` from Kaggle and drop it in `data/`.

## Usage

```bash
python eda.py                        # EDA + plots
python train.py                      # train, tune, evaluate, save best model
python explain.py                    # SHAP explanations on best model
python predict.py data/new_data.csv  # run inference on new transactions
```

## Pipeline

Each model is wrapped in a full `imblearn.Pipeline`:

```
RobustScaler → SMOTE → Model
```

No data leakage — scaling and resampling happen inside the pipeline, so cross-validation is clean.

### What happens in `train.py`:

1. Load and split data (stratified 80/20)
2. Build pipelines for Logistic Regression, Random Forest, XGBoost
3. `RandomizedSearchCV` with 3-fold stratified CV on each, optimizing for average precision
4. Evaluate at default threshold (0.5) and at the optimal threshold from the PR curve
5. Save best pipeline + metadata to `models/`

### Threshold optimization

The default 0.5 threshold is garbage for imbalanced problems. I sweep over all thresholds from the precision-recall curve and pick the one that maximizes F1. Makes a huge difference — especially for Logistic Regression.

### SHAP

`explain.py` runs TreeExplainer on the best model and generates:
- Beeswarm summary (which features push toward fraud)
- Bar plot (mean |SHAP| per feature)
- Waterfall for an individual fraud case

## Project Structure

```
├── eda.py              # exploratory data analysis
├── train.py            # pipeline + tuning + evaluation
├── evaluate.py         # metrics and plotting functions
├── explain.py          # SHAP explainability
├── predict.py          # inference on new data
├── data/               # put creditcard.csv here
├── models/             # saved pipeline + metadata
├── figures/            # all generated plots
└── requirements.txt
```

## Results

| Model               | AP Score | Optimal Threshold | F1 (Fraud) |
|---------------------|----------|-------------------|------------|
| Logistic Regression | ~0.75    | ~0.01             | ~0.60      |
| Random Forest       | ~0.87    | ~0.40             | ~0.85      |
| XGBoost             | ~0.85    | ~0.10             | ~0.80      |

(exact numbers depend on the random search — these are ballpark)

## Built With

- scikit-learn, XGBoost, imbalanced-learn
- SHAP for model explainability
- pandas, matplotlib, seaborn
