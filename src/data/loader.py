import os
import pandas as pd
import numpy as np
from src.utils.logger import setup_logger
from src.utils.config import get_config

logger = setup_logger("data_loader")
config = get_config()

def generate_mock_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generates synthetic credit dataset for testing and execution without download."""
    logger.info(f"Generating {n_samples} mock applicant samples...")
    np.random.seed(config.random_state)
    
    df = pd.DataFrame({
        'SK_ID_CURR': range(100000, 100000 + n_samples),
        'TARGET': np.random.choice([0, 1], size=n_samples, p=[0.92, 0.08]),
        'NAME_CONTRACT_TYPE': np.random.choice(['Cash loans', 'Revolving loans'], size=n_samples),
        'CODE_GENDER': np.random.choice(['M', 'F'], size=n_samples),
        'FLAG_OWN_CAR': np.random.choice(['Y', 'N'], size=n_samples),
        'FLAG_OWN_REALTY': np.random.choice(['Y', 'N'], size=n_samples),
        'CNT_CHILDREN': np.random.poisson(0.5, size=n_samples),
        'AMT_INCOME_TOTAL': np.random.lognormal(11.5, 0.5, size=n_samples),
        'AMT_CREDIT': np.random.exponential(500000, size=n_samples) + 50000,
        'AMT_ANNUITY': np.random.exponential(25000, size=n_samples) + 2000,
        'DAYS_BIRTH': -np.random.randint(7300, 25000, size=n_samples),
        'DAYS_EMPLOYED': -np.random.randint(100, 15000, size=n_samples),
        'EXT_SOURCE_1': np.random.uniform(0.1, 0.9, size=n_samples),
        'EXT_SOURCE_2': np.random.uniform(0.05, 0.95, size=n_samples),
        'EXT_SOURCE_3': np.random.uniform(0.01, 0.85, size=n_samples)
    })
    
    # Intentionally add some missing values
    df.loc[df.sample(frac=0.1).index, 'EXT_SOURCE_1'] = np.nan
    df.loc[df.sample(frac=0.05).index, 'EXT_SOURCE_3'] = np.nan
    df.loc[df.sample(frac=0.03).index, 'AMT_ANNUITY'] = np.nan
    
    return df

def load_dataset(file_name: str) -> pd.DataFrame:
    """
    Loads raw CSV dataset from data_dir. Falls back to mock data if not found.
    """
    data_path = os.path.join(config.data_dir, file_name)
    
    if os.path.exists(data_path):
        logger.info(f"Loading dataset from: {data_path}")
        return pd.read_csv(data_path)
    else:
        logger.warning(f"File {file_name} not found at {config.data_dir}. Falling back to simulated dataset.")
        return generate_mock_data()

def aggregate_bureau_records(bureau_df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function to aggregate historical records from credit bureau.
    """
    logger.info("Aggregating bureau histories...")
    agg_bureau = bureau_df.groupby('SK_ID_CURR').agg({
        'SK_ID_BUREAU': 'count',
        'DAYS_CREDIT': ['min', 'max', 'mean'],
        'AMT_CREDIT_SUM': ['sum', 'mean', 'max']
    })
    
    # Flatten columns
    agg_bureau.columns = ['_'.join(col).strip() for col in agg_bureau.columns.values]
    agg_bureau.rename(columns={'SK_ID_BUREAU_count': 'BUREAU_LOAN_COUNT'}, inplace=True)
    return agg_bureau.reset_index()
