"""
Health Check Routes
"""
from flask import Blueprint, jsonify, current_app
from datetime import datetime

bp = Blueprint('health', __name__, url_prefix='')

@bp.route('/', methods=['GET'])
def root():
    """Root endpoint - API info"""
    return jsonify({
        'status': 'operational',
        'service': 'Skincare Product Recommendation API',
        'version': '2.0.0',
        'timestamp': datetime.now().isoformat(),
        'endpoints': {
            'GET /health': 'Health status',
            'GET /docs': 'API documentation',
            'POST /api/recommend': 'Get recommendations',
            'GET /api/categories': 'Get available categories',
            'GET /api/model-info': 'Get model information'
        }
    }), 200

@bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    from models.recommendation import model
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model.is_trained,
        'model_type': 'Ensemble (KNN + Random Forest)',
        'data_available': model.data is not None
    }), 200

@bp.route('/docs', methods=['GET'])
def docs():
    """API documentation"""
    return jsonify({
        'title': 'Skincare Recommendation API',
        'version': '2.0.0',
        'description': 'ML-powered skincare product recommendation engine',
        'documentation': {
            'POST /api/recommend': {
                'description': 'Get personalized product recommendations',
                'request_body': {
                    'skin_type': 'string (e.g., "oily", "dry", "normal")',
                    'concern_1': 'string (primary concern)',
                    'concern_2': 'string (secondary concern)',
                    'concern_3': 'string (tertiary concern)',
                    'top_n': 'integer (1-50, default: 10)'
                },
                'response': {
                    'status': 'success',
                    'recommendations': [{
                        'rank': 'int',
                        'label': 'string',
                        'brand': 'string',
                        'name': 'string',
                        'price': 'float',
                        'confidence': 'float (0-1)'
                    }]
                }
            },
            'GET /api/categories': {
                'description': 'Get available skin types and concerns',
                'response': {
                    'status': 'success',
                    'categories': {
                        'skin type': ['list of skin types'],
                        'concern': ['list of concerns 1'],
                        'concern 2': ['list of concerns 2'],
                        'concern 3': ['list of concerns 3']
                    }
                }
            },
            'GET /api/model-info': {
                'description': 'Get model performance and details',
                'response': {
                    'status': 'success',
                    'model_info': {
                        'model_type': 'string',
                        'accuracy': 'float',
                        'precision': 'float',
                        'recall': 'float',
                        'f1_score': 'float'
                    }
                }
            }
        }
    }), 200
