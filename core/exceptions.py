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


class InvalidFileFormatException(HairMeException):
    """Raised when uploaded file format is invalid"""

    def __init__(self, message: str = "지원하지 않는 파일 형식입니다. (jpg, jpeg, png, webp만 가능)"):
        super().__init__(message)
