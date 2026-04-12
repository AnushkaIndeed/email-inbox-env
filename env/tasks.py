from abc import ABC, abstractmethod
from typing import List
from .models import Email, Action


def safe_score(raw: float) -> float:
    """Map raw [0,1] score to strictly open (0,1) interval."""
    return round(0.001 + 0.998 * raw, 4)


class Task(ABC):
    @abstractmethod
    def evaluate(self, emails: List[Email], actions: List[Action]) -> float:
        pass

    @abstractmethod
    def grade_step(self, email: Email, action: Action) -> float:
        pass

    @abstractmethod
    def get_description(self) -> str:
        pass




class SpamDetectionTask(Task):

    def get_description(self) -> str:
        return "Detect and properly classify spam emails from the inbox"

    def grade_step(self, email: Email, action: Action) -> float:
        if email.is_spam:
            return 0.85 if action.action_type == "delete" else 0.15
        else:
            return 0.15 if action.action_type == "delete" else 0.50

    def evaluate(self, emails: List[Email], actions: List[Action]) -> float:
        if not emails:
            return 0.20
        correct = sum(
            1 for email, action in zip(emails, actions)
            if (email.is_spam and action.action_type == "delete")
            or (not email.is_spam and action.action_type != "delete")
        )
        return safe_score(correct / len(emails))


class ImportantEmailTask(Task):

    def get_description(self) -> str:
        return "Identify and prioritize important emails for user attention"

    def grade_step(self, email: Email, action: Action) -> float:
        if email.is_important:
            return 0.85 if action.action_type == "classify" else 0.15
        else:
            return 0.50 if action.action_type != "classify" else 0.15

    def evaluate(self, emails: List[Email], actions: List[Action]) -> float:
        if not emails:
            return 0.20
        correct = sum(
            1 for email, action in zip(emails, actions)
            if (email.is_important and action.action_type == "classify")
            or (not email.is_important and action.action_type != "classify")
        )
        return safe_score(correct / len(emails))


class InboxOrganizationTask(Task):

    def get_description(self) -> str:
        return "Organize emails into appropriate folders"

    def grade_step(self, email: Email, action: Action) -> float:
        if action.action_type == "delete" and email.is_spam:
            return 0.85
        elif action.action_type == "classify" and email.is_important:
            return 0.85
        elif action.action_type == "classify":
            return 0.60
        else:
            return 0.30

    def evaluate(self, emails: List[Email], actions: List[Action]) -> float:
        if not emails:
            return 0.20
        correct = sum(
            1 for email, action in zip(emails, actions)
            if action.action_type in ["move", "archive", "classify"]
        )
        return safe_score(correct / len(emails))