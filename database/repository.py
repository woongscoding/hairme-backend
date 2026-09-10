"""Abstract repository interface for analysis data storage"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class AnalysisRepository(ABC):
    """
    Abstract base class for analysis data repositories

    DynamoDB is the only implementation; the interface is kept so business
    logic stays decoupled from the storage layer.
    """

    @abstractmethod
    def save_analysis(
        self,
        image_hash: str,
        analysis_result: Dict[str, Any],
        processing_time: float,
        detection_method: str,
        mp_features: Optional[Any] = None,
    ) -> Optional[str]:
        """
        Save analysis result to database

        Args:
            image_hash: SHA256 hash of the image
            analysis_result: Analysis result dictionary
            processing_time: Processing time in seconds
            detection_method: Detection method used
            mp_features: MediaPipe features (optional)

        Returns:
            Analysis ID (UUID string) if successful, None otherwise
        """
        pass

    @abstractmethod
    def get_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve analysis result by ID

        Args:
            analysis_id: Analysis record ID

        Returns:
            Analysis data dict or None if not found
        """
        pass

    @abstractmethod
    def save_feedback(
        self,
        analysis_id: str,
        style_index: int,
        feedback: str,
        naver_clicked: bool,
    ) -> bool:
        """
        Save user feedback for a specific style recommendation

        Args:
            analysis_id: Analysis record ID
            style_index: Style index (1, 2, or 3)
            feedback: Feedback value ('good' or 'bad')
            naver_clicked: Whether user clicked Naver search link

        Returns:
            bool: True if successful, False otherwise
        """
        pass

    @abstractmethod
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """
        Get aggregated feedback statistics

        Returns:
            Dictionary with statistics data
        """
        pass


def get_repository() -> AnalysisRepository:
    """
    Factory function returning the DynamoDB repository implementation.

    Returns:
        AnalysisRepository: DynamoDB repository instance

    Example:
        >>> repo = get_repository()
        >>> analysis_id = repo.save_analysis(...)
    """
    from database.dynamodb_repository import DynamoDBAnalysisRepository

    return DynamoDBAnalysisRepository()
