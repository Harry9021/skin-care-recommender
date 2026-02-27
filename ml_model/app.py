"""Skincare Recommendation API - Main Flask application"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from flask import Flask, jsonify
from flask_cors import CORS
import logging
import os
from pathlib import Path
from datetime import datetime

from config.settings import active_config
from utils.logger import setup_logger
from utils.errors import SkincareException
from models.recommendation import RecommendationModel, model
from routes import health, recommendations, auth

# Setup logging
log_dir = os.path.dirname(active_config.LOG_FILE)
if log_dir and not os.path.exists(log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)

logger = setup_logger(
    'skincare_api',
    log_level=getattr(logging, active_config.LOG_LEVEL),
    log_file=active_config.LOG_FILE
)

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# CORS setup
CORS(app, resources={
    r"/api/*": {
        "origins": active_config.CORS_ORIGINS,
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": False
    }
})

# Register blueprints
app.register_blueprint(health.bp)
app.register_blueprint(recommendations.bp)
app.register_blueprint(auth.bp)

# Error handlers
@app.errorhandler(SkincareException)
def handle_skincare_exception(err):
    """Handle custom skincare exceptions"""
    return jsonify({
        'status': 'error',
        'error_type': err.__class__.__name__,
        'message': str(err),
        'timestamp': datetime.now().isoformat()
    }), err.status_code if hasattr(err, 'status_code') else 500

@app.errorhandler(404)
def handle_not_found(err):
    """Handle 404 errors"""
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found',
        'timestamp': datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
def handle_internal_error(err):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(err)}")
    return jsonify({
        'status': 'error',
        'message': 'Internal server error',
        'timestamp': datetime.now().isoformat()
    }), 500

# Load model on startup
_is_initialized = False

def init_app():
    """Initialize app with model"""
    global _is_initialized
    if _is_initialized:
        return

    import os as os_module
    current_cwd = os_module.getcwd()
    logger.info(f"Current working directory: {current_cwd}")
    
    logger.info("Initializing Skincare Recommendation API...")
    logger.info(f"Environment: {active_config.__class__.__name__}")
    
    # Load the model
    model_path = active_config.MODEL_PATH
    logger.info(f"Loading model from {model_path}...")
    logger.info(f"Model path exists: {os.path.exists(model_path)}")
    
    if os.path.exists(model_path):
        if model.load(model_path):
            logger.info("[OK] Model loaded successfully")
            logger.info(f"  - Model accuracy: {model.model_metrics.get('accuracy', 'N/A')}")
            logger.info(f"  - Training timestamp: {model.model_metrics.get('timestamp', 'N/A')}")
        else:
            logger.warning(f"[!] Failed to load model from {model_path}")
            logger.info("  Model will be available after training")
    else:
        logger.warning(f"[!] Model file not found at {model_path}")
        logger.info(f"  Expected path: {model_path}")
        logger.info("  Model will be available after training")
    
    # Always try to load dataset for categories endpoint
    dataset_path = active_config.DATASET_PATH
    logger.info(f"Attempting to load dataset from {dataset_path}")
    logger.info(f"Dataset path exists: {os.path.exists(dataset_path)}")
    
    if os.path.exists(dataset_path):
        try:
            logger.info(f"Loading dataset from {dataset_path}...")
            model.load_dataset(dataset_path)
            logger.info(f"[OK] Dataset loaded: {len(model.data)} rows")
        except Exception as e:
            logger.warning(f"[!] Could not load dataset: {str(e)}")
            logger.error(f"Dataset loading error: {type(e).__name__}: {str(e)}")
    else:
        logger.warning(f"[!] Dataset not found at {dataset_path}")
        # Try to find it in ml_model directory
        fallback_path = os.path.join(os.path.dirname(__file__), 'to_be_use_dataset.csv')
        logger.info(f"Trying fallback path: {fallback_path}")
        if os.path.exists(fallback_path):
            try:
                logger.info(f"Loading dataset from fallback path {fallback_path}...")
                model.load_dataset(fallback_path)
                logger.info(f"[OK] Dataset loaded from fallback: {len(model.data)} rows")
            except Exception as e:
                logger.warning(f"[!] Fallback also failed: {str(e)}")
    
    logger.info("[OK] API initialized successfully!\n")
    _is_initialized = True

# Ensure initialization also runs in serverless imports (e.g., Vercel),
# where __main__ block is not executed.
try:
    init_app()
except Exception as e:
    logger.error(f"Startup initialization failed during import: {str(e)}")

# ==================== MAIN ENTRY POINT ====================

if __name__ == '__main__':
    try:
        # Initialize app
        init_app()
        
        # Display startup information
        print("\n" + "="*80)
        print("SKINCARE RECOMMENDATION SYSTEM - BACKEND API")
        print("="*80)
        print(f"\nAPI Server: http://localhost:5000")
        print("\nAvailable Endpoints:")
        print("  GET  /              - Health check and API info")
        print("  GET  /health        - Health status")
        print("  POST /api/recommend - Get product recommendations")
        print("  GET  /api/categories - Get available input options")
        print("  GET  /api/model-info - Get model details")
        print("\nGoogle OAuth Endpoints:")
        print("  POST /api/auth/google         - Login with Google")
        print("  POST /api/auth/verify-token   - Verify JWT token")
        print("  POST /api/auth/refresh-token  - Refresh JWT token")
        print("\nPress CTRL+C to stop the server")
        print("\n" + "="*80 + "\n")
        
        # Run Flask app
        app.run(
            host=os.getenv('FLASK_HOST', '0.0.0.0'),
            port=int(os.getenv('FLASK_PORT', 5000)),
            debug=False
        )
        
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        print(f"\n[ERROR] Server startup failed!")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
