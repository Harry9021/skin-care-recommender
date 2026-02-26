"""
Authentication Routes
Google OAuth and JWT token management
"""
from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta, timezone
import jwt
import requests

from utils.logger import setup_logger
from utils.errors import AuthenticationException
from config.settings import active_config
from middleware.auth_middleware import require_auth

logger = setup_logger(__name__)

bp = Blueprint('auth', __name__, url_prefix='/api/auth')


def _build_access_token(user_info):
    """Create signed JWT access token for authenticated user."""
    now = datetime.now(timezone.utc)
    payload = {
        'user_id': user_info.get('sub') or user_info.get('email'),
        'email': user_info.get('email'),
        'name': user_info.get('name'),
        'picture': user_info.get('picture'),
        'iat': int(now.timestamp()),
        'exp': int((now + timedelta(seconds=active_config.JWT_EXPIRATION)).timestamp())
    }
    return jwt.encode(payload, active_config.JWT_SECRET, algorithm=active_config.JWT_ALGORITHM)


@bp.route('/google', methods=['POST'])
def google_oauth():
    """
    Exchange a Google ID token for an app JWT.

    Request Body:
        - token: str (Google ID token from frontend)
    """
    try:
        data = request.get_json()
        if not data or 'token' not in data:
            raise AuthenticationException("Missing 'token' field")

        google_token = data['token']
        if not isinstance(google_token, str) or not google_token.strip():
            raise AuthenticationException("Invalid Google token")

        verify_url = "https://oauth2.googleapis.com/tokeninfo"
        response = requests.get(
            verify_url,
            params={'id_token': google_token},
            timeout=10
        )

        if response.status_code != 200:
            logger.warning("Google token verification failed")
            raise AuthenticationException("Google token verification failed")

        token_info = response.json()

        # Validate token audience against configured client id
        if active_config.GOOGLE_CLIENT_ID and token_info.get('aud') != active_config.GOOGLE_CLIENT_ID:
            raise AuthenticationException("Token client mismatch")

        email_verified = str(token_info.get('email_verified', 'false')).lower() == 'true'
        if not token_info.get('email') or not email_verified:
            raise AuthenticationException("Google account email not verified")

        access_token = _build_access_token(token_info)

        return jsonify({
            'status': 'success',
            'token': access_token,
            'token_type': 'Bearer',
            'expires_in': active_config.JWT_EXPIRATION,
            'user': {
                'id': token_info.get('sub'),
                'name': token_info.get('name'),
                'email': token_info.get('email'),
                'picture': token_info.get('picture')
            }
        }), 200

    except AuthenticationException as e:
        return jsonify({
            'status': 'error',
            'message': e.message
        }), 401
    except requests.RequestException as e:
        logger.error(f"OAuth network error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Unable to verify Google token'
        }), 503
    except Exception as e:
        logger.error(f"OAuth error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Authentication failed'
        }), 500


@bp.route('/verify-token', methods=['POST'])
@require_auth
def verify_token():
    """Verify JWT token validity and return user payload."""
    return jsonify({
        'status': 'success',
        'valid': True,
        'user': {
            'id': getattr(request, 'user_id', None),
            'email': getattr(request, 'user_email', None)
        }
    }), 200


@bp.route('/refresh-token', methods=['POST'])
@require_auth
def refresh_token():
    """Issue a fresh JWT from a valid existing JWT."""
    try:
        payload = {
            'sub': getattr(request, 'user_id', None),
            'email': getattr(request, 'user_email', None),
            'name': None,
            'picture': None
        }
        new_token = _build_access_token(payload)
        return jsonify({
            'status': 'success',
            'token': new_token,
            'token_type': 'Bearer',
            'expires_in': active_config.JWT_EXPIRATION
        }), 200
    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to refresh token'
        }), 500
