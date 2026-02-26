"""
Authentication Middleware
Handles JWT token verification and authorization
"""
from functools import wraps
from flask import request, jsonify
import jwt
from datetime import datetime

from utils.logger import setup_logger
from utils.errors import AuthenticationException, AuthorizationException
from config.settings import active_config

logger = setup_logger(__name__)

def require_auth(f):
    """
    Decorator to require valid JWT token for endpoint access
    
    Usage:
        @app.route('/protected')
        @require_auth
        def protected_endpoint():
            return jsonify({'status': 'success'})
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            return jsonify({
                'status': 'error',
                'message': 'Missing Authorization header'
            }), 401
        
        try:
            # Extract token
            parts = auth_header.split()
            if len(parts) != 2 or parts[0].lower() != 'bearer':
                raise AuthenticationException("Invalid Authorization header format")
            
            token = parts[1]
            
            # Verify token
            payload = jwt.decode(
                token,
                active_config.JWT_SECRET,
                algorithms=[active_config.JWT_ALGORITHM]
            )
            
            # Store user info in request
            request.user_id = payload.get('user_id')
            request.user_email = payload.get('email')
            
            return f(*args, **kwargs)
            
        except jwt.ExpiredSignatureError:
            logger.warning("Expired token attempted")
            return jsonify({
                'status': 'error',
                'message': 'Token expired'
            }), 401
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': 'Invalid token'
            }), 401
        except AuthenticationException as e:
            return jsonify({
                'status': 'error',
                'message': e.message
            }), 401
        except Exception as e:
            logger.error(f"Auth error: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': 'Authentication failed'
            }), 500
    
    return decorated_function

def optional_auth(f):
    """
    Decorator for optional authentication
    Verifies token if provided, but doesn't require it
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        request.user_id = None
        request.user_email = None
        
        if auth_header:
            try:
                parts = auth_header.split()
                if len(parts) == 2 and parts[0].lower() == 'bearer':
                    token = parts[1]
                    payload = jwt.decode(
                        token,
                        active_config.JWT_SECRET,
                        algorithms=[active_config.JWT_ALGORITHM]
                    )
                    request.user_id = payload.get('user_id')
                    request.user_email = payload.get('email')
            except Exception as e:
                logger.debug(f"Optional auth token could not be verified: {str(e)}")
        
        return f(*args, **kwargs)
    
    return decorated_function

def require_role(required_role):
    """
    Decorator to require specific user role
    
    Usage:
        @app.route('/admin')
        @require_auth
        @require_role('admin')
        def admin_endpoint():
            return jsonify({'status': 'success'})
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not hasattr(request, 'user_role'):
                return jsonify({
                    'status': 'error',
                    'message': 'User role not available'
                }), 401
            
            if request.user_role != required_role:
                logger.warning(f"Unauthorized access attempt: {request.user_role} tried to access {required_role} endpoint")
                return jsonify({
                    'status': 'error',
                    'message': f'This endpoint requires {required_role} role'
                }), 403
            
            return f(*args, **kwargs)
        
        return decorated_function
    
    return decorator
