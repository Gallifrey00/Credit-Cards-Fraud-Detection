import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, auc,
    average_precision_score
)

plt.style.use('seaborn-v0_8-whitegrid')


def plot_confusion_matrix(y_true, y_pred, title='', save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Legit', 'Fraud'],
                yticklabels=['Legit', 'Fraud'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title or 'Confusion Matrix')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close()


def plot_pr_curves(y_true, scores_dict, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, y_score in scores_dict.items():
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax.plot(rec, prec, linewidth=2, label=f'{name} (AP={ap:.3f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall')
    ax.legend(loc='best')
    ax.set_xlim([0, 1.02])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curves(y_true, scores_dict, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, y_score in scores_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC')
    ax.legend(loc='lower right')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close()


def plot_feature_importance(model, feature_names, top_n=15, save_path=None):
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
    elif hasattr(model, 'coef_'):
        imp = np.abs(model.coef_[0])
    else:
        return

    idx = np.argsort(imp)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(idx)), imp[idx], color='steelblue')
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close()


def plot_threshold_analysis(y_true, y_scores, model_name, save_path=None):
    prec, rec, thresholds = precision_recall_curve(y_true, y_scores)
    f1 = np.where((prec + rec) == 0, 0, 2 * prec * rec / (prec + rec))

    # drop the last point (precision_recall_curve adds an extra one)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, prec[:-1], label='Precision', linewidth=2)
    ax.plot(thresholds, rec[:-1], label='Recall', linewidth=2)
    ax.plot(thresholds, f1[:-1], label='F1', linewidth=2, linestyle='--')

    best_idx = np.argmax(f1[:-1])
    best_t = thresholds[best_idx]
    ax.axvline(best_t, color='gray', linestyle=':', alpha=0.7,
               label=f'Best threshold ({best_t:.3f})')

    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title(f'Threshold Analysis — {model_name}')
    ax.legend(loc='best')
    ax.set_xlim([0, 1])
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close()


def print_report(y_true, y_pred, name):
    print(f'\n{"=" * 50}')
    print(f'  {name}')
    print(f'{"=" * 50}')
    print(classification_report(y_true, y_pred,
                                target_names=['Legit', 'Fraud']))
