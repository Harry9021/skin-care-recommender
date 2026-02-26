"""
Custom Exceptions
Define application-specific exceptions for better error handling
"""

class SkincareException(Exception):
    """Base exception for skincare recommendation system"""
    def __init__(self, message, status_code=500, error_code=None):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or self.__class__.__name__
        super().__init__(self.message)


class ModelNotInitializedException(SkincareException):
    """Raised when model is not initialized"""
    def __init__(self, message="Model not initialized. Please wait for system startup."):
        super().__init__(message, status_code=503, error_code="MODEL_NOT_READY")


class InvalidInputException(SkincareException):
    """Raised when input validation fails"""
    def __init__(self, message, details=None):
        super().__init__(message, status_code=400, error_code="INVALID_INPUT")
        self.details = details or {}


class CategoryNotFoundException(SkincareException):
    """Raised when invalid category is provided"""
    def __init__(self, category_type, value, available=None):
        message = f"Invalid {category_type}: '{value}'"
        super().__init__(message, status_code=400, error_code="INVALID_CATEGORY")
        self.category_type = category_type
        self.value = value
        self.available = available


class RecommendationException(SkincareException):
    """Raised during recommendation process"""
    def __init__(self, message, details=None):
        super().__init__(message, status_code=500, error_code="RECOMMENDATION_FAILED")
        self.details = details or {}


class AuthenticationException(SkincareException):
    """Raised for authentication failures"""
    def __init__(self, message="Authentication failed"):
        super().__init__(message, status_code=401, error_code="AUTH_FAILED")


class AuthorizationException(SkincareException):
    """Raised for authorization failures"""
    def __init__(self, message="Not authorized to access this resource"):
        super().__init__(message, status_code=403, error_code="FORBIDDEN")


class DataLoadingException(SkincareException):
    """Raised when data loading fails"""
    def __init__(self, message, filename=None):
        super().__init__(message, status_code=500, error_code="DATA_LOAD_FAILED")
        self.filename = filename
