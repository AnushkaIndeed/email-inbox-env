from typing import Dict, List, Tuple
from enum import Enum


class ActionType(Enum):
    CLASSIFY = "classify"
    ARCHIVE = "archive"
    DELETE = "delete"
    MOVE = "move"


class Grader:
    """Grades agent actions and computes rewards."""

    def __init__(self, spam_penalty: float = -1.0, important_reward: float = 1.0):
        """Initialize grader with reward params."""
        self.spam_penalty = spam_penalty
        self.important_reward = important_reward
        self.correct_classifications = 0
        self.total_classifications = 0

    def grade_action(
        self, action_type: str, email_is_spam: bool, email_is_important: bool
    ) -> float:
        """Grade an action and return reward."""
        reward = 0.0

        if action_type == ActionType.DELETE.value:
            # Reward for deleting spam, penalty for deleting important
            reward = self.spam_penalty if not email_is_spam else 1.0
            if email_is_important:
                reward -= 2.0

        elif action_type == ActionType.CLASSIFY.value:
            # Reward for accurate classification
            if email_is_spam or email_is_important:
                reward = self.important_reward
                self.correct_classifications += 1
            self.total_classifications += 1

        elif action_type == ActionType.ARCHIVE.value:
            # Small reward for archiving processed emails
            reward = 0.1 if not email_is_important else -0.5

        elif action_type == ActionType.MOVE.value:
            # Reward for moving emails to appropriate folder
            reward = 0.5

        return reward

    def compute_metrics(
        self, actions: List[str], email_labels: List[Tuple[bool, bool]]
    ) -> Dict[str, float]:
        correct = sum(
            1 for action, (is_spam, is_imp) in zip(actions, email_labels)
            if (action == "delete" and is_spam) or (action == "classify" and (is_spam or is_imp))
        )
        total = len(actions)
        accuracy = correct / total if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "total_processed": total,
            "correct_actions": correct,
        }
