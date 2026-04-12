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
        # Safe reward range: 0.20 to 0.80
        if email.is_spam:
            return 0.80 if action.action_type == "delete" else 0.20
        else:
            if action.action_type == "delete":
                return 0.20
            return 0.50 # Correct baseline

    def evaluate(self, emails: List[Email], actions: List[Action]) -> float:
        if not emails:
            return 0.20
        correct = 0
        for email, action in zip(emails, actions):
            if email.is_spam and action.action_type == "delete":
                correct += 1
            elif not email.is_spam and action.action_type != "delete":
                correct += 1
        
        raw_score = correct / len(emails)
        # Safe scale and round: 0.2 + 0.6 * raw_score => [0.20, 0.80]
        return round(0.2 + 0.6 * raw_score, 2)


# -------------------- IMPORTANT EMAIL --------------------

class ImportantEmailTask(Task):

    def get_description(self) -> str:
        return "Identify and prioritize important emails for user attention"

    def grade_step(self, email: Email, action: Action) -> float:
        if email.is_important:
            return 0.80 if action.action_type == "classify" else 0.20
        else:
            return 0.50 if action.action_type != "classify" else 0.20

    def evaluate(self, emails: List[Email], actions: List[Action]) -> float:
        if not emails:
            return 0.20
        correct = 0
        for email, action in zip(emails, actions):
            if email.is_important and action.action_type == "classify":
                correct += 1
            elif not email.is_important and action.action_type != "classify":
                correct += 1
        
        raw_score = correct / len(emails)
        return round(0.2 + 0.6 * raw_score, 2)


# -------------------- INBOX ORGANIZATION --------------------

class InboxOrganizationTask(Task):

    def get_description(self) -> str:
        return "Organize emails into appropriate folders"

    def grade_step(self, email: Email, action: Action) -> float:
        # Balanced rewards strictly within (0.20, 0.80)
        if action.action_type == "delete" and email.is_spam:
            return 0.80
        elif action.action_type == "classify" and email.is_important:
            return 0.80
        elif action.action_type == "classify":
            return 0.60
        else:
            return 0.30

    def evaluate(self, emails: List[Email], actions: List[Action]) -> float:
        if not emails:
            return 0.20
        
        correct = 0
        for email, action in zip(emails, actions):
            # For organization, any move/archive/intelligent classify is considered correct
            if action.action_type in ["move", "archive", "classify"]:
                correct += 1
        
        raw_score = correct / len(emails)
        return round(0.2 + 0.6 * raw_score, 2)