"""
Input Validators
Provides validation functions for API inputs
"""
from typing import List, Dict, Any
from .errors import InvalidInputException, CategoryNotFoundException


class InputValidator:
    """Handles input validation"""
    
    @staticmethod
    def validate_recommendation_request(data: Dict[str, Any], encoders: Dict) -> Dict[str, str]:
        """
        Validate recommendation request input
        
        Args:
            data: Request data dictionary
            encoders: Label encoders for valid categories
            
        Returns:
            dict: Validated input data
            
        Raises:
            InvalidInputException: If validation fails
        """
        required_fields = ['skin_type', 'concern_1', 'concern_2', 'concern_3']
        missing = [f for f in required_fields if f not in data]
        
        if missing:
            raise InvalidInputException(
                f"Missing required fields: {', '.join(missing)}",
                details={'required_fields': required_fields, 'received_fields': list(data.keys())}
            )
        
        validated = {}
        
        # Validate skin_type
        if 'skin type' not in encoders:
            raise InvalidInputException("Skin type encoder not available")
        
        skin_types = encoders['skin type'].classes_
        if data['skin_type'] not in skin_types:
            raise CategoryNotFoundException(
                'skin_type',
                data['skin_type'],
                available=sorted(skin_types.tolist())[:5]
            )
        validated['skin_type'] = data['skin_type']
        
        # Validate concerns
        concern_fields = ['concern_1', 'concern_2', 'concern_3']
        concern_encoder_keys = ['concern', 'concern 2', 'concern 3']
        
        for field, encoder_key in zip(concern_fields, concern_encoder_keys):
            if encoder_key not in encoders:
                raise InvalidInputException(f"{encoder_key} encoder not available")
            
            concerns = encoders[encoder_key].classes_
            if data[field] not in concerns:
                raise CategoryNotFoundException(
                    encoder_key,
                    data[field],
                    available=sorted(concerns.tolist())[:5]
                )
            validated[field] = data[field]
        
        # Validate top_n
        top_n = data.get('top_n', 10)
        if not isinstance(top_n, int) or top_n < 1 or top_n > 50:
            raise InvalidInputException(
                "top_n must be an integer between 1 and 50",
                details={'received': top_n, 'valid_range': '1-50'}
            )
        validated['top_n'] = top_n
        
        return validated
    
    @staticmethod
    def validate_pagination(page: int = 1, per_page: int = 10) -> tuple:
        """
        Validate pagination parameters
        
        Returns:
            tuple: (page, per_page)
            
        Raises:
            InvalidInputException: If validation fails
        """
        try:
            page = int(page)
            per_page = int(per_page)
        except (ValueError, TypeError):
            raise InvalidInputException("page and per_page must be integers")
        
        if page < 1:
            raise InvalidInputException("page must be >= 1")
        if per_page < 1 or per_page > 100:
            raise InvalidInputException("per_page must be between 1 and 100")
        
        return page, per_page
