"""Custom exception classes for HairMe Backend"""


class HairMeException(Exception):
    """Base exception for HairMe application"""

    def __init__(self, message: str = "An error occurred"):
        self.message = message
        super().__init__(self.message)


class NoFaceDetectedException(HairMeException):
    """Raised when no face is detected in the uploaded image"""

    def __init__(self, message: str = "얼굴이 감지되지 않았습니다.\n밝은 곳에서 정면 사진을 촬영해주세요."):
        super().__init__(message)


class MultipleFacesException(HairMeException):
    """Raised when multiple faces are detected in the uploaded image"""

    def __init__(self, face_count: int = 2, message: str = None):
        if message is None:
            message = f"{face_count}명의 얼굴이 감지되었습니다.\n한 명만 나온 사진을 업로드해주세요."
        self.face_count = face_count
        super().__init__(message)


class DatabaseException(HairMeException):
    """Raised when database operations fail"""

    def __init__(self, message: str = "데이터베이스 오류가 발생했습니다"):
        super().__init__(message)


class CacheException(HairMeException):
    """Raised when cache operations fail"""

    def __init__(self, message: str = "캐시 오류가 발생했습니다"):
        super().__init__(message)


class MLModelException(HairMeException):
    """Raised when ML model operations fail"""

    def __init__(self, message: str = "ML 모델 오류가 발생했습니다"):
        super().__init__(message)


class GeminiAPIException(HairMeException):
    """Raised when Gemini API operations fail"""

    def __init__(self, message: str = "Gemini API 오류가 발생했습니다"):
        super().__init__(message)


class GeminiRateLimitException(GeminiAPIException):
    """Raised when Gemini API rate limit is exceeded"""

    def __init__(self, message: str = "Gemini API 요청 한도를 초과했습니다. 잠시 후 다시 시도해주세요."):
        super().__init__(message)


class GeminiInvalidResponseException(GeminiAPIException):
    """Raised when Gemini API returns invalid/unparseable response"""

    def __init__(self, message: str = "Gemini API 응답을 파싱할 수 없습니다"):
        super().__init__(message)


class GeminiAuthenticationException(GeminiAPIException):
    """Raised when Gemini API authentication fails"""

    def __init__(self, message: str = "Gemini API 인증에 실패했습니다"):
        super().__init__(message)


class InvalidFileFormatException(HairMeException):
    """Raised when uploaded file format is invalid"""

    def __init__(self, message: str = "지원하지 않는 파일 형식입니다. (jpg, jpeg, png, webp만 가능)"):
        super().__init__(message)


class StyleNotFoundException(MLModelException):
    """Raised when hairstyle is not found in ML model database"""

    def __init__(self, style_name: str, message: str = None):
        if message is None:
            message = f"헤어스타일을 찾을 수 없습니다: {style_name}"
        self.style_name = style_name
        super().__init__(message)


class ModelNotLoadedException(MLModelException):
    """Raised when ML model is not loaded but required"""

    def __init__(self, message: str = "ML 모델이 로드되지 않았습니다"):
        super().__init__(message)


class DynamoDBException(DatabaseException):
    """Raised when DynamoDB operations fail"""

    def __init__(self, message: str = "DynamoDB 오류가 발생했습니다"):
        super().__init__(message)


class DynamoDBConnectionException(DynamoDBException):
    """Raised when cannot connect to DynamoDB"""

    def __init__(self, message: str = "DynamoDB 연결에 실패했습니다"):
        super().__init__(message)


class DynamoDBResourceNotFoundException(DynamoDBException):
    """Raised when DynamoDB table/resource not found"""

    def __init__(self, resource_name: str, message: str = None):
        if message is None:
            message = f"DynamoDB 리소스를 찾을 수 없습니다: {resource_name}"
        self.resource_name = resource_name
        super().__init__(message)


class MediaPipeException(HairMeException):
    """Raised when MediaPipe operations fail"""

    def __init__(self, message: str = "얼굴 분석 중 오류가 발생했습니다"):
        super().__init__(message)


class CircuitBreakerOpenException(HairMeException):
    """Raised when circuit breaker is open (service unavailable)"""

    def __init__(self, service_name: str = "Service", message: str = None):
        if message is None:
            message = f"{service_name} 일시적으로 사용할 수 없습니다. 잠시 후 다시 시도해주세요."
        self.service_name = service_name
        super().__init__(message)
