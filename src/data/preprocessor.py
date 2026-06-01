import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils.logger import setup_logger

logger = setup_logger("preprocessor")

class CreditRiskPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn compliant preprocessor for Credit Risk data modeling.
    Separates categorical and numerical columns, applies imputation, scaling, and encoding.
    """
    def __init__(self):
        self.pipeline = None
        self.num_cols = []
        self.cat_cols = []

    def fit(self, X: pd.DataFrame, y=None):
        # Drop identifiers and target
        features = X.drop(columns=['SK_ID_CURR', 'TARGET'], errors='ignore')
        
        # Categorize features
        self.num_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = features.select_dtypes(exclude=[np.number]).columns.tolist()
        
        logger.info(f"Detected numerical columns ({len(self.num_cols)}): {self.num_cols[:5]}...")
        logger.info(f"Detected categorical columns ({len(self.cat_cols)}): {self.cat_cols[:5]}...")
        
        # Numeric pipeline: Impute with median, scale features
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline: Impute with most frequent, one-hot encode
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combined Preprocessor
        self.pipeline = ColumnTransformer([
            ('num', num_pipeline, self.num_cols),
            ('cat', cat_pipeline, self.cat_cols)
        ])
        
        self.pipeline.fit(features)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        features = X.drop(columns=['SK_ID_CURR', 'TARGET'], errors='ignore')
        if self.pipeline is None:
            raise ValueError("The preprocessor has not been fitted yet!")
        
        # Ensure all columns expected by the fitted pipeline are present at inference
        expected_cols = self.num_cols + self.cat_cols
        missing_cols = [col for col in expected_cols if col not in features.columns]
        if missing_cols:
            # Concatenate missing columns at once to prevent pandas fragmentation PerformanceWarnings
            missing_df = pd.DataFrame(np.nan, index=features.index, columns=missing_cols)
            features = pd.concat([features, missing_df], axis=1)
                
        # Reorder to match the exact fitted column signature
        features = features[expected_cols]
        
        return self.pipeline.transform(features)

    def get_feature_names_out(self) -> list:
        """Helper to get names of columns post one-hot encoding."""
        if self.pipeline is None:
            return []
        
        # Get numeric names
        num_names = self.num_cols
        
        # Get one-hot names
        onehot_transformer = self.pipeline.named_transformers_['cat'].named_steps['encoder']
        cat_names = onehot_transformer.get_feature_names_out(self.cat_cols).tolist()
        
        return num_names + cat_names
