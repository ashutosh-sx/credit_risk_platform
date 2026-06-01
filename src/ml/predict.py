import os
import argparse
import joblib
import pandas as pd
from typing import Dict, Any, Union
from src.utils.logger import setup_logger
from src.utils.config import get_config

logger = setup_logger("predict_pipeline")
config = get_config()

def load_pipeline_artifacts(model_name: str) -> Dict[str, Any]:
    """Loads preprocessor and trained estimator model from serialized path."""
    model_path = os.path.join(config.models_dir, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Please run train.py first.")
    
    logger.info(f"Loading model pipeline artifacts from: {model_path}")
    return joblib.load(model_path)

def score_applicant(data: Union[pd.DataFrame, Dict[str, Any]], model_name: str = "credit_risk_model.joblib") -> pd.DataFrame:
    """
    Scores client credit risk.
    Accepts pandas DataFrame or raw dictionary inputs representing a single client.
    """
    artifacts = load_pipeline_artifacts(model_name)
    preprocessor = artifacts['preprocessor']
    model = artifacts['model']
    
    if isinstance(data, dict):
        df_input = pd.DataFrame([data])
    else:
        df_input = data.copy()
        
    # Preprocess inference payload
    processed_features = preprocessor.transform(df_input)
    
    # Predict default probabilities and binary target labels
    proba = model.predict_proba(processed_features)[:, 1]
    predictions = model.predict(processed_features)
    
    results = pd.DataFrame({
        'SK_ID_CURR': df_input['SK_ID_CURR'] if 'SK_ID_CURR' in df_input.columns else range(len(df_input)),
        'DEFAULT_PROBABILITY': proba,
        'PREDICTED_DEFAULT': predictions
    })
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Risk Inference Engine")
    parser.add_argument("--model_name", type=str, default="credit_risk_model.joblib", help="Model filename under models_dir")
    parser.add_argument("--batch_input", type=str, help="Path to batch inference csv data")
    
    args = parser.parse_args()
    
    if args.batch_input and os.path.exists(args.batch_input):
        logger.info(f"Running batch prediction on: {args.batch_input}")
        batch_df = pd.read_csv(args.batch_input)
        preds = score_applicant(batch_df, args.model_name)
        
        output_path = os.path.join(config.data_dir, "predictions_output.csv")
        preds.to_csv(output_path, index=False)
        logger.info(f"Batch prediction complete! Outputs saved to {output_path}")
    else:
        # Run a simple mock scoring case for diagnostic verification
        logger.info("No input file provided or path does not exist. Running diagnostic sample prediction.")
        sample_client = {
            'SK_ID_CURR': 999999,
            'NAME_CONTRACT_TYPE': 'Cash loans',
            'CODE_GENDER': 'F',
            'FLAG_OWN_CAR': 'Y',
            'FLAG_OWN_REALTY': 'Y',
            'CNT_CHILDREN': 1,
            'AMT_INCOME_TOTAL': 180000.0,
            'AMT_CREDIT': 450000.0,
            'AMT_ANNUITY': 22500.0,
            'DAYS_BIRTH': -12500,
            'DAYS_EMPLOYED': -2000,
            'EXT_SOURCE_1': 0.65,
            'EXT_SOURCE_2': 0.72,
            'EXT_SOURCE_3': 0.58
        }
        try:
            res = score_applicant(sample_client, args.model_name)
            print("\n--- Scoring Diagnostic Results ---")
            print(res.to_string(index=False))
        except Exception as e:
            logger.error(f"Failed to score diagnostic sample: {str(e)}")
            logger.warning("Please ensure train.py was run to generate model artifacts.")
