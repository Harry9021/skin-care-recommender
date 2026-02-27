"""
Configuration Management
Loads settings from environment variables and provides defaults
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Runtime-aware directories
# APP_DIR points to the backend root (ml_model directory) in both local and Vercel.
APP_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = APP_DIR.parent

def resolve_path(path_value: str, default_relative: str) -> str:
    """
    Resolve file paths safely across local/dev and serverless deployments.

    Resolution order:
    1) Absolute path (as-is)
    2) Relative to APP_DIR (ml_model)
    3) Relative to REPO_DIR (project root)
    4) Fallback to APP_DIR/default
    """
    raw = (path_value or default_relative).strip()
    candidate = Path(raw)
    if candidate.is_absolute():
        return str(candidate)

    app_candidate = APP_DIR / candidate
    if app_candidate.exists():
        return str(app_candidate)

    repo_candidate = REPO_DIR / candidate
    if repo_candidate.exists():
        return str(repo_candidate)

    return str(APP_DIR / default_relative)

class Config:
    """Base configuration"""
    # Flask
    DEBUG = False
    TESTING = False
    
    # CORS
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:3001').split(',')
    
    # Server
    HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    PORT = int(os.getenv('FLASK_PORT', 5000))
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/app.log')
    
    # Model and data paths
    MODEL_PATH = resolve_path(os.getenv('MODEL_PATH', ''), 'skincare_model_enhanced.pkl')
    DATASET_PATH = resolve_path(os.getenv('DATASET_PATH', ''), 'to_be_use_dataset.csv')
    DATA_KINDS_PATH = resolve_path(os.getenv('DATA_KINDS_PATH', ''), 'data_kinds.txt')
    
    # Recommendations
    DEFAULT_TOP_N = int(os.getenv('DEFAULT_TOP_N', 10))
    MAX_TOP_N = int(os.getenv('MAX_TOP_N', 50))
    MIN_TOP_N = int(os.getenv('MIN_TOP_N', 1))
    
    # Google OAuth (Add your API key in .env file)
    GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID', '')
    GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET', '')
    
    # JWT
    JWT_SECRET = os.getenv('JWT_SECRET', 'your-secret-key-change-in-production')
    JWT_ALGORITHM = os.getenv('JWT_ALGORITHM', 'HS256')
    JWT_EXPIRATION = int(os.getenv('JWT_EXPIRATION', 86400))  # 24 hours
    
    @staticmethod
    def validate():
        """Validate critical configuration"""
        issues = []
        
        # Warn about missing OAuth config (not critical)
        if not Config.GOOGLE_CLIENT_ID:
            issues.append("WARNING: GOOGLE_CLIENT_ID not set. OAuth will not work.")
        
        if not Config.GOOGLE_CLIENT_SECRET:
            issues.append("WARNING: GOOGLE_CLIENT_SECRET not set. OAuth will not work.")
        
        if Config.JWT_SECRET == 'your-secret-key-change-in-production':
            issues.append("WARNING: Using default JWT_SECRET. Change this in production!")
        
        return issues


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    MODEL_PATH = 'test_model.pkl'
    LOG_LEVEL = 'DEBUG'


# Select config based on environment
ENV = os.getenv('FLASK_ENV', 'development').lower()

if ENV == 'production':
    active_config = ProductionConfig
elif ENV == 'testing':
    active_config = TestingConfig
else:
    active_config = DevelopmentConfig

# Validate configuration on import
config_issues = active_config.validate()
if config_issues:
    for issue in config_issues:
        print(f"[CONFIG] {issue}")
