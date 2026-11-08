"""Custom exception classes for HairMe Backend"""


class HairMeException(Exception):
    """Base exception for HairMe application"""

    def __init__(self, message: str = "An error occurred"):
        self.message = message
        super().__init__(self.message)


class NoFaceDetectedException(HairMeException):
    """Raised when no face is detected in the uploaded image"""

    def __init__(
        self,
        message: str = "얼굴이 감지되지 않았습니다.\n밝은 곳에서 정면 사진을 촬영해주세요.",
    ):
        super().__init__(message)


class MultipleFacesException(HairMeException):
    """Raised when multiple faces are detected in the uploaded image"""

    def __init__(self, face_count: int = 2, message: str = None):
        if message is None:
            message = f"{face_count}명의 얼굴이 감지되었습니다.\n한 명만 나온 사진을 업로드해주세요."
        self.face_count = face_count
        super().__init__(message)


class FullBodyNotDetectedException(HairMeException):
    """Raised when shoulders and hips are not both visible in the photo"""

    def __init__(
        self,
        message: str = "어깨와 골반이 보이지 않아요.\n허리 아래까지 나오도록 정면에서 찍어주세요.",
    ):
        super().__init__(message)


class NotFrontalPoseException(HairMeException):
    """Raised when the body is turned too far sideways to measure proportions"""

    def __init__(
        self,
        message: str = "몸이 옆으로 돌아가 있어요.\n카메라를 향해 정면으로 서서 찍어주세요.",
    ):
        super().__init__(message)


class InvalidFileFormatException(HairMeException):
    """Raised when uploaded file format is invalid"""

    def __init__(
        self,
        message: str = "지원하지 않는 파일 형식입니다. (jpg, jpeg, png, webp만 가능)",
    ):
        super().__init__(message)
