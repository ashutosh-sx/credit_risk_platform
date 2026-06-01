#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exploratory Data Analysis (EDA) Script
Converted from eda.ipynb
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure source modules are importable from notebooks directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import setup_logger

logger = setup_logger("eda_script")

def run_eda():
    sns.set_theme(style="darkgrid", palette="muted")
    plt.rcParams["figure.figsize"] = (12, 6)

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "application_train.csv")
    
    if not os.path.exists(data_path):
        logger.warning(f"Dataset not found at {data_path}. Creating a dummy dataframe for demonstration.")
        # Generate dummy data for illustration
        np.random.seed(42)
        n_samples = 1000
        df = pd.DataFrame({
            'SK_ID_CURR': range(100000, 100000 + n_samples),
            'TARGET': np.random.choice([0, 1], size=n_samples, p=[0.92, 0.08]),
            'NAME_CONTRACT_TYPE': np.random.choice(['Cash loans', 'Revolving loans'], size=n_samples),
            'CODE_GENDER': np.random.choice(['M', 'F'], size=n_samples),
            'FLAG_OWN_CAR': np.random.choice(['Y', 'N'], size=n_samples),
            'CNT_CHILDREN': np.random.poisson(0.5, size=n_samples),
            'AMT_INCOME_TOTAL': np.random.lognormal(11.5, 0.5, size=n_samples),
            'AMT_CREDIT': np.random.exponential(500000, size=n_samples) + 50000,
            'DAYS_BIRTH': -np.random.randint(7300, 25000, size=n_samples),
            'EXT_SOURCE_1': np.random.uniform(0.1, 0.9, size=n_samples),
            'EXT_SOURCE_2': np.random.uniform(0.05, 0.95, size=n_samples),
            'EXT_SOURCE_3': np.random.uniform(0.01, 0.85, size=n_samples)
        })
        # Introduce some missing values
        df.loc[df.sample(frac=0.15).index, 'EXT_SOURCE_1'] = np.nan
        df.loc[df.sample(frac=0.05).index, 'EXT_SOURCE_3'] = np.nan
    else:
        df = pd.read_csv(data_path)

    print(f"\n--- Dataset Shape: {df.shape} ---")
    print(df.head())

    # Target distribution
    target_counts = df['TARGET'].value_counts(normalize=True) * 100
    print("\n--- Target Distribution (%) ---")
    print(target_counts)

    # Plot Target Distribution
    plt.figure(figsize=(6, 5))
    sns.countplot(x='TARGET', data=df)
    plt.title('Distribution of Target Variable (0 = Repaid, 1 = Defaulted)')
    plt.ylabel('Count')
    output_dir = os.path.join(os.path.dirname(__file__), "..", "documents")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "target_distribution.png"))
    plt.close()
    logger.info("Saved target_distribution.png in documents/ directory.")

    # Scores Correlation
    cols_scores = ['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    corr = df[cols_scores].corr()
    print("\n--- Correlation Matrix ---")
    print(corr)

    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.3f', vmin=-1, vmax=1)
    plt.title('Correlation Matrix of External Score Ratings')
    plt.savefig(os.path.join(output_dir, "scores_correlation.png"))
    plt.close()
    logger.info("Saved scores_correlation.png in documents/ directory.")

    # Score distribution KDE
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, col in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):
        sns.kdeplot(data=df, x=col, hue='TARGET', common_norm=False, fill=True, alpha=0.4, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel('Score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "score_distributions.png"))
    plt.close()
    logger.info("Saved score_distributions.png in documents/ directory. EDA Script Execution Complete!")

if __name__ == "__main__":
    run_eda()
