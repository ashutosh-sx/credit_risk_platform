import os
import argparse
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from src.data.loader import load_dataset
from src.data.preprocessor import CreditRiskPreprocessor
from src.utils.logger import setup_logger
from src.utils.config import get_config

logger = setup_logger("train_pipeline")
config = get_config()

def run_training(data_file: str, model_name: str):
    logger.info("Initializing credit risk model training...")
    
    # 1. Load Data
    df = load_dataset(data_file)
    
    if 'TARGET' not in df.columns:
        raise KeyError("Dataset must contain the 'TARGET' column indicating credit default.")
        
    X = df.drop(columns=['TARGET'])
    y = df['TARGET']
    
    # 2. Train Test Split
    X_train, X_val, y_train, y_val = train_test_split(
        df, y, 
        test_size=config.test_size, 
        random_state=config.random_state, 
        stratify=y
    )
    logger.info(f"Split completed. Train shape: {X_train.shape}, Validation shape: {X_val.shape}")
    
    # 3. Preprocessing
    logger.info("Fitting data preprocessor...")
    preprocessor = CreditRiskPreprocessor()
    preprocessor.fit(X_train)
    
    X_train_proc = preprocessor.transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    
    # 4. Model Definition & Training
    logger.info("Initializing ML classifier model...")
    # Defaulting to RandomForest for standard availability, easily replaceable with LightGBM
    clf = RandomForestClassifier(
        n_estimators=config.n_estimators, 
        max_depth=10, 
        random_state=config.random_state, 
        class_weight='balanced',
        n_jobs=-1
    )
    
    logger.info("Fitting classifier...")
    clf.fit(X_train_proc, y_train)
    
    # 5. Evaluate training accuracy
    train_score = clf.score(X_train_proc, y_train)
    val_score = clf.score(X_val_proc, y_val)
    logger.info(f"Training Score (Accuracy): {train_score:.4f}")
    logger.info(f"Validation Score (Accuracy): {val_score:.4f}")
    
    # 6. Save Model Artifacts
    os.makedirs(config.models_dir, exist_ok=True)
    model_path = os.path.join(config.models_dir, model_name)
    
    artifacts = {
        'preprocessor': preprocessor,
        'model': clf,
        'features': preprocessor.get_feature_names_out()
    }
    
    joblib.dump(artifacts, model_path)
    logger.info(f"Successfully serialized model artifacts to {model_path}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Risk Model Training Pipeline")
    parser.add_argument("--data_file", type=str, default="application_train.csv", help="Name of dataset file")
    parser.add_argument("--model_name", type=str, default="credit_risk_model.joblib", help="Filename of saved model artifact")
    
    args = parser.parse_args()
    run_training(args.data_file, args.model_name)
