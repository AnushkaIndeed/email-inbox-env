from abc import ABC, abstractmethod
from typing import List
from .models import Email, Action


class Task(ABC):

    @abstractmethod
    def evaluate(self, emails: List[Email], actions: List[Action]) -> float:
        """Evaluate agent performance on task."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get task description."""
        pass


class SpamDetectionTask(Task):
    """Task: Identify and remove spam emails."""

    def get_description(self) -> str:
        return "Detect and properly classify spam emails from the inbox"

    def evaluate(self, emails: List[Email], actions: List[Action]) -> float:
        """Score based on spam detection accuracy."""
        correct = 0
        for email, action in zip(emails, actions):
            if email.is_spam and action.action_type == "delete":
                correct += 1
            elif not email.is_spam and action.action_type != "delete":
                correct += 1
        
        return correct / len(emails) if emails else 0.0


class ImportantEmailTask(Task):
    """Task: Prioritize and flag important emails."""

    def get_description(self) -> str:
        return "Identify and prioritize important emails for user attention"

    def evaluate(self, emails: List[Email], actions: List[Action]) -> float:
        """Score based on important email identification."""
        correct = 0
        for email, action in zip(emails, actions):
            if email.is_important and action.action_type == "classify":
                correct += 1
            elif not email.is_important and action.action_type != "classify":
                correct += 1
        
        return correct / len(emails) if emails else 0.0


class InboxOrganizationTask(Task):
    """Task: Organize inbox into folders based on content."""

    def get_description(self) -> str:
        return "Organize emails into appropriate folders (work, personal, etc.)"

    def evaluate(self, emails: List[Email], actions: List[Action]) -> float:
        """Score based on appropriate folder organization."""
        if not emails:
            return 0.0
        
        correct = 0
        for email, action in zip(emails, actions):
            # Reward any organizational action (move or archive)
            if action.action_type in ["move", "archive"]:
                correct += 1
        
        return correct / len(emails)
