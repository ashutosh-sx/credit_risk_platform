import os
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    classification_report,
    confusion_matrix
)
from src.utils.logger import setup_logger
from src.utils.config import get_config

logger = setup_logger("evaluate_pipeline")
config = get_config()

def evaluate_model(data_file: str, model_name: str):
    logger.info("Initializing model evaluation...")
    
    # 1. Load Model Pipeline Artifacts
    model_path = os.path.join(config.models_dir, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Please train a model first.")
        
    artifacts = joblib.load(model_path)
    preprocessor = artifacts['preprocessor']
    model = artifacts['model']
    
    # 2. Load Evaluation Data
    data_path = os.path.join(config.data_dir, data_file)
    if not os.path.exists(data_path):
        logger.warning(f"Evaluation dataset not found at {data_path}. Creating mock evaluation dataset.")
        # Load dataset internally generates mock if application_train.csv is absent
        from src.data.loader import load_dataset
        df = load_dataset("application_train.csv")
    else:
        df = pd.read_csv(data_path)
        
    if 'TARGET' not in df.columns:
        raise KeyError("Evaluation dataset must contain the target 'TARGET' column.")
        
    X = df.drop(columns=['TARGET'])
    y = df['TARGET']
    
    # 3. Transform features and score probabilities
    X_proc = preprocessor.transform(X)
    probs = model.predict_proba(X_proc)[:, 1]
    preds = model.predict(X_proc)
    
    # 4. Compute Metrics
    roc_auc = roc_auc_score(y, probs)
    
    precision, recall, _ = precision_recall_curve(y, probs)
    pr_auc = auc(recall, precision)
    
    logger.info("Evaluation Metrics:")
    logger.info(f"  * ROC-AUC score: {roc_auc:.5f}")
    logger.info(f"  * PR-AUC (Average Precision): {pr_auc:.5f}")
    
    print("\n--- Classification Report ---")
    print(classification_report(y, preds))
    
    print("--- Confusion Matrix ---")
    cm = confusion_matrix(y, preds)
    print(f"True Negative (Repaid): {cm[0][0]} | False Positive (False Alarm): {cm[0][1]}")
    print(f"False Negative (Missed Default): {cm[1][0]} | True Positive (Defaulted): {cm[1][1]}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Risk Evaluation Diagnostics")
    parser.add_argument("--data_file", type=str, default="application_train.csv", help="Filename of evaluation dataset")
    parser.add_argument("--model_name", type=str, default="credit_risk_model.joblib", help="Filename of trained model artifact")
    
    args = parser.parse_args()
    evaluate_model(args.data_file, args.model_name)
