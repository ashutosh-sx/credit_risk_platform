import numpy as np
import pandas as pd
from typing import Dict, Any

def days_to_years(days: int) -> float:
    """
    Converts age representation in negative integer days to absolute years.
    Commonly used in Home Credit datasets where DAYS_BIRTH is negative.
    """
    return abs(days) / 365.25

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Flattens a nested dictionary, e.g., for config or metrics logging.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def format_percentage(val: float) -> str:
    """Formats raw floats to formatted percent strings."""
    return f"{val * 100:.2f}%"

def calculate_woe(df: pd.DataFrame, feature: str, target: str) -> pd.DataFrame:
    """
    Helper function to calculate Weight of Evidence (WoE) for a credit risk feature.
    Useful for credit scorecards and exploratory risk analyses.
    """
    eps = 1e-6
    total_goods = (df[target] == 0).sum()
    total_bads = (df[target] == 1).sum()
    
    grouped = df.groupby(feature)[target].agg(
        total_count='count',
        bad_count='sum'
    )
    grouped['good_count'] = grouped['total_count'] - grouped['bad_count']
    
    grouped['good_dist'] = grouped['good_count'] / total_goods + eps
    grouped['bad_dist'] = grouped['bad_count'] / total_bads + eps
    
    grouped['WOE'] = np.log(grouped['good_dist'] / grouped['bad_dist'])
    grouped['IV'] = (grouped['good_dist'] - grouped['bad_dist']) * grouped['WOE']
    
    return grouped
