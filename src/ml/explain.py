import numpy as np
import pandas as pd
from typing import Dict, Any, List

def calculate_feature_contributions(client_data: Dict[str, Any], training_data: pd.DataFrame = None) -> List[Dict[str, Any]]:
    """
    Computes explainable AI feature contributions (like SHAP / LIME values)
    by calculating the applicant's normalized deviation from demographic averages.
    
    Positive contributions represent features increasing default risk.
    Negative contributions represent features reducing default risk.
    """
    # 1. Standardize population baselines and risk vectors
    # Features, their population averages (or standard profiles), standard deviations, and risk direction
    # risk_direction: +1 means higher values increase default risk, -1 means higher values decrease default risk.
    baselines = {
        'EXT_SOURCE_1': {'mean': 0.50, 'std': 0.20, 'risk_dir': -1, 'label': 'External Credit Score 1'},
        'EXT_SOURCE_2': {'mean': 0.51, 'std': 0.21, 'risk_dir': -1, 'label': 'External Credit Score 2'},
        'EXT_SOURCE_3': {'mean': 0.49, 'std': 0.19, 'risk_dir': -1, 'label': 'External Credit Score 3'},
        'AMT_INCOME_TOTAL': {'mean': 168000.0, 'std': 80000.0, 'risk_dir': -1, 'label': 'Total Income'},
        'AMT_CREDIT': {'mean': 599000.0, 'std': 400000.0, 'risk_dir': 1, 'label': 'Requested Credit Amount'},
        'AMT_ANNUITY': {'mean': 27000.0, 'std': 14000.0, 'risk_dir': 1, 'label': 'Annuity Amount'},
        'CNT_CHILDREN': {'mean': 0.41, 'std': 0.72, 'risk_dir': 1, 'label': 'Number of Children'},
        'DAYS_BIRTH': {'mean': -16000, 'std': 4300, 'risk_dir': -1, 'label': 'Applicant Age (DAYS_BIRTH)'},
        'DAYS_EMPLOYED': {'mean': -2300, 'std': 2300, 'risk_dir': -1, 'label': 'Employment Duration'}
    }
    
    # 2. Extract training values dynamically if available
    if training_data is not None:
        for feat in baselines.keys():
            if feat in training_data.columns:
                baselines[feat]['mean'] = training_data[feat].mean()
                baselines[feat]['std'] = training_data[feat].std() or 1.0

    contributions = []
    
    # 3. Calculate client deviation and contribution
    for feat, meta in baselines.items():
        client_val = client_data.get(feat, None)
        if client_val is None or pd.isna(client_val):
            continue
            
        mean = meta['mean']
        std = meta['std']
        risk_dir = meta['risk_dir']
        
        # Calculate raw deviation
        dev = client_val - mean
        
        # Standardized deviation
        std_dev = dev / std
        
        # Calculate contribution: deviation scaled by risk direction
        # E.g., if credit requested is higher (+1 dir) than mean, contribution is positive (adds risk).
        # E.g., if EXT_SOURCE_2 score is higher (-1 dir) than mean, contribution is negative (reduces risk).
        contribution = std_dev * risk_dir * 1.5
        
        # Clip contribution range for clean plotting
        contribution = np.clip(contribution, -5.0, 5.0)
        
        # Formulate a plain-english explanation
        if contribution > 0.5:
            reason = f"Significantly higher risk: {meta['label']} is in a riskier tier compared to the average profile."
        elif contribution < -0.5:
            reason = f"Significantly lower risk: {meta['label']} represents strong creditworthiness relative to population average."
        else:
            reason = f"Neutral risk: {meta['label']} matches the baseline client profile."
            
        contributions.append({
            'Feature': feat,
            'Label': meta['label'],
            'Client Value': client_val,
            'Population Average': mean,
            'Contribution': contribution,
            'Description': reason
        })
        
    # Sort contributions by magnitude of impact (high risk drivers first)
    contributions = sorted(contributions, key=lambda x: abs(x['Contribution']), reverse=True)
    return contributions
