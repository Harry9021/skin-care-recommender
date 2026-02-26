"""
Configuration Management
Loads settings from environment variables and provides defaults
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the base directory (project root)
BASE_DIR = Path(__file__).parent.parent.parent
ML_MODEL_DIR = BASE_DIR / 'ml_model'

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
    
    # Model - use absolute paths
    _model_path = os.getenv('MODEL_PATH', 'ml_model/skincare_model_enhanced.pkl')
    if os.path.isabs(_model_path):
        MODEL_PATH = _model_path
    else:
        MODEL_PATH = str(BASE_DIR / _model_path)
    
    _dataset_path = os.getenv('DATASET_PATH', 'ml_model/to_be_use_dataset.csv')
    if os.path.isabs(_dataset_path):
        DATASET_PATH = _dataset_path
    else:
        DATASET_PATH = str(BASE_DIR / _dataset_path)
    
    _data_kinds_path = os.getenv('DATA_KINDS_PATH', 'ml_model/data_kinds.txt')
    if os.path.isabs(_data_kinds_path):
        DATA_KINDS_PATH = _data_kinds_path
    else:
        DATA_KINDS_PATH = str(BASE_DIR / _data_kinds_path)
    
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
