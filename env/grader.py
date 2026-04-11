from typing import Dict, List, Tuple
from .models import Email, Action


class Grader:
    """Computes metrics for email inbox environment."""

    def __init__(self):
        """Initialize grader."""
        self.correct_classifications = 0
        self.total_processed = 0

    def compute_metrics(
        self, emails: List[Email], actions: List[Action], task_eval_score: float
    ) -> Dict[str, float]:
        """Compute metrics for the episode."""
        total = len(emails)
        
        return {
            "accuracy": task_eval_score,  # Use task-specific evaluation as baseline accuracy
            "total_processed": total,
            "reward_sum": sum(actions[i].confidence for i in range(len(actions))), # Example aggregate
        }
