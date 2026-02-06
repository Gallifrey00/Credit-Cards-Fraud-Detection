import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.style.use('seaborn-v0_8-whitegrid')


def run_eda(filepath='data/creditcard.csv'):
    os.makedirs('figures', exist_ok=True)

    df = pd.read_csv(filepath)
    print(f'shape: {df.shape}')
    print(f'frauds: {df["Class"].sum()} / {len(df)} ({df["Class"].mean():.4%})')
    print()
    print(df.describe().round(2))

    # class distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df['Class'].value_counts()
    ax.bar(['Legit', 'Fraud'], counts.values, color=['steelblue', 'indianred'])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 1000, str(v), ha='center', fontsize=11)
    ax.set_ylabel('Count')
    ax.set_title('Transaction Class Distribution')
    plt.tight_layout()
    fig.savefig('figures/class_dist.png', dpi=150)
    plt.close()

    # amount — fraud vs legit
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df[df['Class'] == 0]['Amount'], bins=50, alpha=0.7,
                 color='steelblue', edgecolor='white')
    axes[0].set_title('Legit — Amount')
    axes[0].set_xlabel('Amount')
    axes[0].set_xlim(0, 2500)

    axes[1].hist(df[df['Class'] == 1]['Amount'], bins=50, alpha=0.7,
                 color='indianred', edgecolor='white')
    axes[1].set_title('Fraud — Amount')
    axes[1].set_xlabel('Amount')
    plt.tight_layout()
    fig.savefig('figures/amount_dist.png', dpi=150)
    plt.close()

    # transactions over time
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df[df['Class'] == 0]['Time'], bins=100, alpha=0.5,
            label='Legit', color='steelblue')
    ax.hist(df[df['Class'] == 1]['Time'], bins=100, alpha=0.7,
            label='Fraud', color='indianred')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Count')
    ax.set_title('Transactions Over Time')
    ax.legend()
    plt.tight_layout()
    fig.savefig('figures/time_dist.png', dpi=150)
    plt.close()

    # top correlated features
    corr_with_class = df.drop('Class', axis=1).corrwith(df['Class']).abs()
    top = corr_with_class.nlargest(10)

    fig, ax = plt.subplots(figsize=(8, 5))
    top.sort_values().plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('|Correlation| with Fraud')
    ax.set_title('Top 10 Features Correlated with Fraud')
    plt.tight_layout()
    fig.savefig('figures/top_correlations.png', dpi=150)
    plt.close()

    # boxplots for the top 4
    top_4 = corr_with_class.nlargest(4).index.tolist()
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, feat in enumerate(top_4):
        sns.boxplot(data=df, x='Class', y=feat, ax=axes[i],
                    palette=['steelblue', 'indianred'])
        axes[i].set_xticklabels(['Legit', 'Fraud'])
        axes[i].set_title(feat)
    plt.tight_layout()
    fig.savefig('figures/feature_boxplots.png', dpi=150)
    plt.close()

    print('\nplots saved to figures/')
    return df


if __name__ == '__main__':
    run_eda()
