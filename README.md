# Credit Card Fraud Detection

Binary classification on the [Kaggle credit card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Heavily imbalanced (~0.17% fraud), so the focus is on precision-recall rather than accuracy.

## Setup

```bash
pip install -r requirements.txt
```

Download `creditcard.csv` from Kaggle and drop it in `data/`.

## Usage

```bash
python eda.py       # exploratory analysis + plots
python train.py     # trains models, saves figures, prints results
```

## Approach

- Features V1-V28 are PCA-transformed (from the dataset), Amount and Time are scaled with RobustScaler
- SMOTE to handle class imbalance during training
- Three models compared: Logistic Regression, Random Forest, XGBoost
- Evaluated on precision-recall (AP score) since accuracy is useless here — a model that predicts "legit" 100% of the time gets 99.8% accuracy

## Project Structure

```
├── eda.py              # exploratory data analysis
├── train.py            # model training + comparison
├── evaluate.py         # metrics, plots, confusion matrices
├── data/               # put creditcard.csv here
├── figures/            # generated plots
└── requirements.txt
```

## Built With

- scikit-learn
- XGBoost
- imbalanced-learn (SMOTE)
- pandas, matplotlib, seaborn
