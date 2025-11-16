"""Abstract repository interface for analysis data storage"""

import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union


class AnalysisRepository(ABC):
    """
    Abstract base class for analysis data repositories

    This interface allows seamless switching between MySQL and DynamoDB
    without changing business logic code.
    """

    @abstractmethod
    def save_analysis(
        self,
        image_hash: str,
        analysis_result: Dict[str, Any],
        processing_time: float,
        detection_method: str,
        mp_features: Optional[Any] = None
    ) -> Optional[Union[int, str]]:
        """
        Save analysis result to database

        Args:
            image_hash: SHA256 hash of the image
            analysis_result: Analysis result dictionary
            processing_time: Processing time in seconds
            detection_method: Detection method used
            mp_features: MediaPipe features (optional)

        Returns:
            Record ID if successful (int for MySQL, str for DynamoDB), None otherwise
        """
        pass

    @abstractmethod
    def get_analysis(self, analysis_id: Union[int, str]) -> Optional[Dict[str, Any]]:
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
        analysis_id: Union[int, str],
        style_index: int,
        feedback: str,
        naver_clicked: bool
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
    Factory function to get the appropriate repository implementation
    based on environment configuration

    Returns:
        AnalysisRepository: DynamoDB or MySQL repository instance

    Example:
        >>> repo = get_repository()
        >>> analysis_id = repo.save_analysis(...)
    """
    use_dynamodb = os.getenv('USE_DYNAMODB', 'false').lower() == 'true'

    if use_dynamodb:
        from database.dynamodb_repository import DynamoDBAnalysisRepository
        return DynamoDBAnalysisRepository()
    else:
        from database.mysql_repository import MySQLAnalysisRepository
        return MySQLAnalysisRepository()
