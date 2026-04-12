from abc import ABC, abstractmethod
from typing import List, Tuple
from .models import Email, Action


class Task(ABC):

    @abstractmethod
    def evaluate(self, emails: List[Email], actions: List[Action]) -> float:
        """Evaluate agent performance on task."""
        pass

    @abstractmethod
    def grade_step(self, email: Email, action: Action) -> float:
        """Calculate reward for a single step within this task."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get task description."""
        pass


class SpamDetectionTask(Task):
    """Task: Identify and remove spam emails."""

    def get_description(self) -> str:
        return "Detect and properly classify spam emails from the inbox"

    def grade_step(self, email: Email, action: Action) -> float:
        """
        Reward within (0, 1) to satisfy strict validator constraints.
        Correct: 0.99
        Incorrect: 0.01
        """
        if email.is_spam:
            return 0.99 if action.action_type == "delete" else 0.01
        else:
            if action.action_type == "delete":
                return 0.01 # Incorrect (deleted important/valid)
            return 0.10 # Correct baseline (kept valid email)

    def evaluate(self, emails: List[Email], actions: List[Action]) -> float:
        """Score based on spam detection accuracy (scaled for validator constraints)."""
        if not emails:
            return 0.01
        correct = 0
        for email, action in zip(emails, actions):
            if email.is_spam and action.action_type == "delete":
                correct += 1
            elif not email.is_spam and action.action_type != "delete":
                correct += 1
        
        raw_score = correct / len(emails)
        return 0.01 + 0.98 * raw_score


class ImportantEmailTask(Task):
    """Task: Prioritize and flag important emails."""

    def get_description(self) -> str:
        return "Identify and prioritize important emails for user attention"

    def grade_step(self, email: Email, action: Action) -> float:
        """
        Reward within (0, 1) to satisfy strict validator constraints.
        Correct: 0.99
        Incorrect: 0.01
        """
        if email.is_important:
            return 0.99 if action.action_type == "classify" else 0.01
        else:
            return 0.10 if action.action_type != "classify" else 0.01

    def evaluate(self, emails: List[Email], actions: List[Action]) -> float:
        """Score based on important email identification (scaled for validator constraints)."""
        if not emails:
            return 0.01
        correct = 0
        for email, action in zip(emails, actions):
            if email.is_important and action.action_type == "classify":
                correct += 1
            elif not email.is_important and action.action_type != "classify":
                correct += 1
        
        raw_score = correct / len(emails)
        return 0.01 + 0.98 * raw_score


class InboxOrganizationTask(Task):
    """Task: Organize inbox into folders based on content."""

    def get_description(self) -> str:
        return "Organize emails into appropriate folders (work, personal, etc.)"

    def grade_step(self, email: Email, action: Action) -> float:
        """
        Reward within (0, 1) to satisfy strict validator constraints.
        Correct: 0.90
        Incorrect: 0.01
        """
        if action.action_type in ["move", "archive"]:
            return 0.90
        return 0.01

    def evaluate(self, emails: List[Email], actions: List[Action]) -> float:
        """Score based on appropriate folder organization (scaled for validator constraints)."""
        if not emails:
            return 0.01
        
        correct = 0
        for email, action in zip(emails, actions):
            if action.action_type in ["move", "archive"]:
                correct += 1
        
        raw_score = correct / len(emails)
        return 0.01 + 0.98 * raw_score
