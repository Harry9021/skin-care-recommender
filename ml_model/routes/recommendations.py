"""
Recommendation Routes
Main API endpoints for recommendations
"""
from flask import Blueprint, request, jsonify, current_app
from datetime import datetime

from utils.logger import setup_logger
from utils.validators import InputValidator
from utils.errors import SkincareException
from models.recommendation import model
from middleware.auth_middleware import require_auth

logger = setup_logger(__name__)

bp = Blueprint('recommendations', __name__, url_prefix='/api')

def transform_categories_to_readable(numeric_categories):
    """
    Transform numeric category values to readable format with labels
    
    Args:
        numeric_categories: dict with column names as keys and lists of numeric values
        
    Returns:
        dict with readable category names and labels
    """
    transformed = {}
    
    def _parse_numeric(value):
        """Safely parse value to int when it represents a numeric code."""
        try:
            if isinstance(value, (int, float)):
                return int(value)
            if isinstance(value, str) and value.strip() != "":
                return int(float(value.strip()))
        except (ValueError, TypeError):
            return None
        return None

    for col, values in numeric_categories.items():
        if col == 'skin type':
            # Keep `value` as the original encoder class. This ensures frontend
            # submits exactly what backend validators/model encoders expect.
            options = []
            for val in values:
                raw_value = str(val)
                numeric_val = _parse_numeric(val)
                readable = SKIN_TYPE_MAPPING.get(numeric_val, raw_value) if numeric_val is not None else raw_value
                options.append({
                    'value': raw_value,
                    'label': readable
                })
            transformed[col] = sorted(
                options,
                key=lambda x: (_parse_numeric(x['value']) is None, _parse_numeric(x['value']) if _parse_numeric(x['value']) is not None else x['value'])
            )
        
        elif col in ['concern', 'concern 2', 'concern 3']:
            # Keep raw encoder class in `value`; use mapping only for display `label`.
            options = []
            for val in values:
                raw_value = str(val)
                numeric_val = _parse_numeric(val)
                if numeric_val is not None and 0 <= numeric_val <= 33:
                    readable = CONCERNS_MAPPING.get(numeric_val, raw_value)
                else:
                    readable = raw_value
                options.append({
                    'value': raw_value,
                    'label': readable
                })
            transformed[col] = sorted(
                options,
                key=lambda x: (_parse_numeric(x['value']) is None, _parse_numeric(x['value']) if _parse_numeric(x['value']) is not None else x['value'])
            )
        else:
            # For any other columns, pass through as-is
            transformed[col] = values
    
    return transformed

# Category mappings from data_kinds.txt
SKIN_TYPE_MAPPING = {
    0: 'All',
    1: 'Normal',
    2: 'Dry',
    3: 'Oily',
    4: 'Combination',
    5: 'Sensitive'
}

CONCERNS_MAPPING = {
    0: 'Anti-Pollution',
    1: 'Tan Removal',
    2: 'Dryness',
    3: 'Deep Nourishment',
    4: 'Blackheads and Whiteheads',
    5: 'Oil Control',
    6: 'Fine Lines and Wrinkles',
    7: 'Uneven Skin Tone',
    8: 'Dark Spots',
    9: 'Dark Circles',
    10: 'Skin Tightening',
    11: 'Under Eye Concern',
    12: 'Skin Inflammation',
    13: 'General Care',
    14: 'Redness',
    15: 'Skin Sagging',
    16: 'Lightening',
    17: 'Sun Protection',
    18: 'Pigmentation',
    19: 'Blackheads Removal',
    20: 'Oily Skin',
    21: 'Anti-Ageing',
    22: 'Hydration',
    23: 'Dull Skin',
    24: 'Uneven Texture',
    25: 'Irregular Textures',
    26: 'Pore Minimizing and Blurring',
    27: 'Excess Oil',
    28: 'Daily Use',
    29: 'Dullness',
    30: 'Anti Acne Scarring',
    31: 'Softening and Smoothening',
    32: 'Acne or Blemishes',
    33: 'Pore Care'
}

@bp.route('/recommend', methods=['POST'])
@require_auth
def recommend():
    """
    Get skincare product recommendations
    
    Request Body:
        - skin_type: str
        - concern_1: str
        - concern_2: str
        - concern_3: str
        - top_n: int (optional, default: 10)
    
    Returns:
        - recommendations: list of products with confidence scores
    """
    # Check if model is trained for this endpoint
    if not model.is_trained:
        logger.warning("Model not trained when /recommend was called")
        logger.warning(f"Model data available: {model.data is not None}")
        logger.warning(f"Label encoders available: {len(model.label_encoders) if model.label_encoders else 0}")
        return jsonify({
            'status': 'error',
            'message': 'Model not trained yet. Please wait for model to load...'
        }), 503
    
    try:
        if not request.is_json:
            return jsonify({
                'status': 'error',
                'message': 'Request must be JSON'
            }), 400
        
        # Validate input
        validated = InputValidator.validate_recommendation_request(
            request.json,
            model.label_encoders
        )
        
        # Get recommendations
        recommendations = model.recommend(
            validated['skin_type'],
            validated['concern_1'],
            validated['concern_2'],
            validated['concern_3'],
            top_n=validated['top_n']
        )
        
        return jsonify({
            'status': 'success',
            'count': len(recommendations),
            'recommendations': recommendations.to_dict('records'),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except SkincareException as e:
        logger.warning(f"Validation error: {e.message}")
        return jsonify({
            'status': 'error',
            'error_code': e.error_code,
            'message': e.message,
            'details': e.details if hasattr(e, 'details') else None
        }), e.status_code
        
    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Internal server error',
            'error': str(e)
        }), 500

@bp.route('/categories', methods=['GET'])
@require_auth
def categories():
    """
    Get available categories (skin types and concerns)
    
    Returns:
        - categories: dict with available options for each field
    """
    try:
        from config.settings import active_config
        import os
        
        logger.info("=== Categories endpoint called ===")
        logger.info(f"Model trained: {model.is_trained}")
        logger.info(f"Label encoders count: {len(model.label_encoders) if model.label_encoders else 0}")
        logger.info(f"Model.data is None: {model.data is None}")
        
        categories = {}
        
        # PRIORITY 1: If model has label encoders (from trained pickle), use those
        if model.label_encoders and len(model.label_encoders) > 0:
            logger.info("Using label encoders from trained model")
            for col in model.feature_columns:
                if col in model.label_encoders:
                    classes_list = sorted(model.label_encoders[col].classes_.tolist())
                    categories[col] = classes_list
                    logger.info(f"  {col}: {len(classes_list)} options")
        
        # FALLBACK: If no label encoders, load from dataset
        if not categories or len(categories) < len(model.feature_columns):
            logger.info("Label encoders incomplete, attempting to load from dataset")
            
            if model.data is None:
                try:
                    dataset_path = active_config.DATASET_PATH
                    logger.info(f"Dataset path: {dataset_path}")
                    logger.info(f"Dataset exists: {os.path.exists(dataset_path)}")
                    
                    if not os.path.exists(dataset_path):
                        return jsonify({
                            'status': 'error',
                            'message': f'Dataset file not found at {dataset_path}'
                        }), 503
                    
                    model.load_dataset(dataset_path)
                    logger.info(f"Dataset loaded: {len(model.data)} rows")
                    
                except Exception as e:
                    logger.error(f"Failed to load dataset: {str(e)}")
                    return jsonify({
                        'status': 'error',
                        'message': f'Failed to load dataset: {str(e)}'
                    }), 503
            
            # Extract categories from loaded data
            if model.data is not None:
                logger.info("Extracting categories from dataset")
                for col in model.feature_columns:
                    if col in model.data.columns:
                        col_values = sorted(model.data[col].dropna().astype(str).unique().tolist())
                        # If we don't have this column from encoders, add it now
                        if col not in categories:
                            categories[col] = col_values
                        logger.info(f"  {col}: {len(col_values)} options")
        
        # Summary statistics
        summary = {
            'total_skin_types': len(categories.get('skin type', [])),
            'total_concerns': len(categories.get('concern', [])),
            'total_products': len(model.data) if model.data is not None else 0,
            'model_trained': model.is_trained
        }
        
        if not categories:
            logger.error("No categories found after processing")
            return jsonify({
                'status': 'error',
                'message': 'No categories available'
            }), 500
        
        # Transform numeric values to readable labels
        readable_categories = transform_categories_to_readable(categories)
        
        logger.info(f"Returning {len(readable_categories)} category groups")
        return jsonify({
            'status': 'success',
            'categories': readable_categories,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Categories error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': 'Failed to retrieve categories',
            'error': str(e)
        }), 500

@bp.route('/model-info', methods=['GET'])
@require_auth
def model_info():
    """
    Get model information and performance metrics
    
    Returns:
        - model_type: str
        - metrics: model performance metrics
        - feature_importance: relative importance of features
    """
    try:
        if not model.is_trained:
            return jsonify({
                'status': 'error',
                'message': 'Model not trained'
            }), 503
        
        return jsonify({
            'status': 'success',
            'model_info': {
                'model_type': 'Ensemble (KNN + Random Forest)',
                'is_trained': model.is_trained,
                'n_features': len(model.feature_columns),
                'feature_columns': model.feature_columns,
                'n_products': len(model.data) if model.data is not None else 0,
                'metrics': model.model_metrics,
                'feature_importance': model.feature_importance
            },
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to retrieve model info',
            'error': str(e)
        }), 500

