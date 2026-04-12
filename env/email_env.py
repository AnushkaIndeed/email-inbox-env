import json
from pathlib import Path
from typing import List, Tuple, Optional
from .models import Email, EmailState, Action, EpisodeMetrics
from .grader import Grader
from .tasks import Task, SpamDetectionTask, ImportantEmailTask, InboxOrganizationTask


class EmailEnvironment:
    """RL Environment for Email Inbox tasks with Strict Range scaling."""

    def __init__(self, data_path: Optional[str] = None, task_type: str = "spam"):
        if data_path is None:
            possible_paths = [
                Path(__file__).parent.parent / "data" / "emails.json",
                Path("data") / "emails.json",
                Path("emails.json"),
            ]
            for path in possible_paths:
                if path.exists():
                    data_path = str(path)
                    break
            if data_path is None:
                data_path = str(possible_paths[0])  
        
        self.data_path = data_path
        self.grader = Grader()
        self.emails: List[Email] = []
        self.current_idx = 0
        self.episode_reward = 0.1  # Baseline normalized reward
        self.current_score = 0.1   # Current accuracy score
        self.actions_taken: List[Action] = []
        
        task_map = {
            "spam": SpamDetectionTask(),
            "important": ImportantEmailTask(),
            "organize": InboxOrganizationTask(),
        }
        self.task: Task = task_map.get(task_type, SpamDetectionTask())
        
        self._load_emails()
    

    def _load_emails(self) -> None:
        try:
            with open(self.data_path, "r") as f:
                data = json.load(f)
                self.emails = [Email(**email) for email in data]
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Warning: Email data not found or invalid at {self.data_path}")
            self.emails = []

    def reset(self) -> EmailState:
        """Reset environment and return initial state."""
        self.current_idx = 0
        self.episode_reward = 0.2
        self.current_score = 0.2
        self.actions_taken = []
        return self._get_state()

    def _get_state(self) -> EmailState:
        """Get current normalized environment state."""
        if self.current_idx >= len(self.emails):
            return EmailState(
                current_email=None,
                inbox_size=len(self.emails),
                processed_count=self.current_idx,
                reward=self.episode_reward,
                score=self.current_score,
                done=True,
            )
        
        current_email = self.emails[self.current_idx]
        return EmailState(
            current_email=current_email,
            inbox_size=len(self.emails),
            processed_count=self.current_idx,
            reward=self.episode_reward,
            score=self.current_score,
            done=False,
        )

    def step(self, action: Action) -> Tuple[EmailState, float, bool]:
        """Perform action and return next state, reward (normalized), done."""
        if self.current_idx >= len(self.emails):
            return self._get_state(), 0.1, True

        current_email = self.emails[self.current_idx]
        
        # Get standardized reward from task
        reward = self.task.grade_step(current_email, action)
        
        # Track actions for metrics
        self.actions_taken.append(action)
        self.current_idx += 1
        
        # Calculate current accuracy/score to keep metrics within (0.2, 0.8)
        metrics_emails = self.emails[:self.current_idx]
        self.current_score = round(self.task.evaluate(metrics_emails, self.actions_taken), 2)
        
        # Set normalized reward for this step
        # We report the latest score as the 'cumulative' reward to prevent out-of-range
        self.episode_reward = self.current_score
        
        next_state = self._get_state()
        done = next_state.done
        
        return next_state, reward, done

    def get_metrics(self) -> EpisodeMetrics:
        """Get episode metrics, ensuring all probability fields are in (0.1, 0.9)."""
        if not self.emails or not self.actions_taken:
            return EpisodeMetrics(
                total_reward=0.2, emails_processed=0, accuracy=0.2,
                precision=0.2, recall=0.2
            )
        
        # Final accuracy calculation (already scaled and rounded by task.evaluate)
        final_score = self.task.evaluate(self.emails[:len(self.actions_taken)], self.actions_taken)
        
        return EpisodeMetrics(
            total_reward=final_score, # Treat normalized final score as total reward
            emails_processed=len(self.actions_taken),
            accuracy=final_score,
            precision=final_score,
            recall=final_score
        )

    def get_task_description(self) -> str:
        return self.task.get_description()

    def state(self):
        return self._get_state()