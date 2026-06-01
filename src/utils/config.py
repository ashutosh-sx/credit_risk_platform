import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

@dataclass(frozen=True)
class AppConfig:
    # Environment settings
    environment: str = os.getenv("ENVIRONMENT", "local")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # LLM Settings
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    
    # Paths (Resolved at runtime)
    data_dir: str = os.getenv("DATA_DIR", "data")
    models_dir: str = os.getenv("MODELS_DIR", "models")
    
    # Database Settings
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///data/credit_risk.db")
    
    # Model Hyperparameters (Defaults)
    random_state: int = int(os.getenv("RANDOM_STATE", "42"))
    test_size: float = float(os.getenv("TEST_SIZE", "0.2"))
    
    # LightGBM hyperparams
    learning_rate: float = 0.05
    n_estimators: int = 200
    num_leaves: int = 31

_config = AppConfig()

def get_config() -> AppConfig:
    """Returns the global AppConfig instance."""
    return _config
