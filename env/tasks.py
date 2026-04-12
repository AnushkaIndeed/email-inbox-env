from abc import ABC, abstractmethod
from typing import List
from .models import Email, Action


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


# -------------------- SPAM DETECTION --------------------

class SpamDetectionTask(Task):

    def get_description(self) -> str:
        return "Detect and properly classify spam emails from the inbox"

    def grade_step(self, email: Email, action: Action) -> float:
        # Safe reward range: 0.1 to 0.9
        if email.is_spam:
            return 0.9 if action.action_type == "delete" else 0.1
        else:
            if action.action_type == "delete":
                return 0.1
            return 0.6

    def evaluate(self, emails: List[Email], actions: List[Action]) -> float:
        if not emails:
            return 0.2
        correct = 0
        for email, action in zip(emails, actions):
            if email.is_spam and action.action_type == "delete":
                correct += 1
            elif not email.is_spam and action.action_type != "delete":
                correct += 1
        
        raw_score = correct / len(emails)
        # Safe scale: 0.2 + 0.6 * raw_score => [0.2, 0.8]
        return 0.2 + 0.6 * raw_score


# -------------------- IMPORTANT EMAIL --------------------

class ImportantEmailTask(Task):

    def get_description(self) -> str:
        return "Identify and prioritize important emails for user attention"

    def grade_step(self, email: Email, action: Action) -> float:
        if email.is_important:
            return 0.8 if action.action_type == "classify" else 0.2
        else:
            return 0.5 if action.action_type != "classify" else 0.2

    def evaluate(self, emails: List[Email], actions: List[Action]) -> float:
        if not emails:
            return 0.2
        correct = 0
        for email, action in zip(emails, actions):
            if email.is_important and action.action_type == "classify":
                correct += 1
            elif not email.is_important and action.action_type != "classify":
                correct += 1
        
        raw_score = correct / len(emails)
        return 0.2 + 0.6 * raw_score


# -------------------- INBOX ORGANIZATION --------------------

class InboxOrganizationTask(Task):

    def get_description(self) -> str:
        return "Organize emails into appropriate folders"

    def grade_step(self, email: Email, action: Action) -> float:
        # Balanced rewards strictly within (0.2, 0.8)
        if action.action_type == "delete" and email.is_spam:
            return 0.8
        elif action.action_type == "classify" and email.is_important:
            return 0.8
        elif action.action_type == "classify":
            return 0.6
        else:
            return 0.3

    def evaluate(self, emails: List[Email], actions: List[Action]) -> float:
        if not emails:
            return 0.2
        
        correct = 0
        for email, action in zip(emails, actions):
            # For organization, any move/archive/intelligent classify is considered correct
            if action.action_type in ["move", "archive", "classify"]:
                correct += 1
        
        raw_score = correct / len(emails)
        return 0.2 + 0.6 * raw_score