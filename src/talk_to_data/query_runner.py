import os
import sqlite3
import pandas as pd
from src.utils.logger import setup_logger
from src.utils.config import get_config
from src.data.loader import generate_mock_data

logger = setup_logger("query_runner")
config = get_config()

def get_db_connection() -> sqlite3.Connection:
    """
    Establishes and returns a database connection.
    Automatically initializes the database tables if they do not exist yet.
    """
    # Extract DB path from configuration URL (e.g. sqlite:///data/credit_risk.db)
    db_path = config.database_url.replace("sqlite:///", "")
    
    # Handle absolute or relative paths
    if not os.path.isabs(db_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        db_path = os.path.join(base_dir, db_path)
        
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    
    # Initialize schema and data if table 'applications' is missing
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='applications';")
    table_exists = cursor.fetchone()
    
    if not table_exists:
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "application_train.csv")
        columns = [
            'SK_ID_CURR', 'TARGET', 'NAME_CONTRACT_TYPE', 'CODE_GENDER', 
            'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 
            'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 
            'DAYS_BIRTH', 'DAYS_EMPLOYED', 'EXT_SOURCE_1', 
            'EXT_SOURCE_2', 'EXT_SOURCE_3'
        ]
        if os.path.exists(csv_path):
            logger.info("Table 'applications' not found. Real dataset found. Populating from application_train.csv...")
            try:
                df = pd.read_csv(csv_path, usecols=columns)
                # Sample 10,000 stratified rows for optimal SQLite performance
                defaulted = df[df['TARGET'] == 1]
                repaid = df[df['TARGET'] == 0]
                sample_size = min(10000, len(df))
                default_frac = len(defaulted) / len(df)
                n_default = int(sample_size * default_frac)
                n_repaid = sample_size - n_default
                defaulted_sample = defaulted.sample(n=min(n_default, len(defaulted)), random_state=42)
                repaid_sample = repaid.sample(n=min(n_repaid, len(repaid)), random_state=42)
                sampled_df = pd.concat([defaulted_sample, repaid_sample]).sample(frac=1.0, random_state=42)
                sampled_df.to_sql("applications", conn, index=False, if_exists="replace")
                logger.info("Successfully populated 'applications' table with 10,000 real dataset rows!")
            except Exception as e:
                logger.error(f"Failed to populate real dataset: {e}. Falling back to simulated data.")
                df = generate_mock_data(n_samples=500)
                df.to_sql("applications", conn, index=False, if_exists="replace")
        else:
            logger.warning("Table 'applications' not found. Populating with simulated applicant data...")
            df = generate_mock_data(n_samples=500)
            df.to_sql("applications", conn, index=False, if_exists="replace")
            logger.info("Successfully populated 'applications' table in sqlite database!")
            
    return conn

def execute_query(sql_query: str) -> pd.DataFrame:
    """
    Executes a SELECT query on the database.
    To prevent SQL injections, this function enforces read-only operations.
    """
    # Clean and check SQL action
    clean_sql = sql_query.strip().upper()
    
    # Disallow destructive modifications
    forbidden_words = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE"]
    if any(word in clean_sql for word in forbidden_words):
        raise PermissionError(
            f"Execution Blocked: This engine only supports read-only operations. "
            f"Detected forbidden keyword in query."
        )
        
    conn = get_db_connection()
    try:
        logger.info("Executing database query...")
        df = pd.read_sql_query(sql_query, conn)
        return df
    finally:
        conn.close()
