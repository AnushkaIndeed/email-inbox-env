import json
from pathlib import Path
from typing import List, Tuple, Optional
from .models import Email, EmailState, Action, EpisodeMetrics
from .grader import Grader
from .tasks import Task, SpamDetectionTask, ImportantEmailTask, InboxOrganizationTask


class EmailEnvironment:

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
        self.episode_reward = 0.0
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
        except FileNotFoundError:
            print(f"Warning: Email data not found at {self.data_path}")
            self.emails = []

    def reset(self) -> EmailState:
        """Reset environment and return initial state."""
        self.current_idx = 0
        self.episode_reward = 0.0
        self.actions_taken = []
        return self._get_state()

    def _get_state(self) -> EmailState:
        """Get current environment state."""
        if self.current_idx >= len(self.emails):
            return EmailState(
                current_email=None,
                inbox_size=len(self.emails),
                processed_count=self.current_idx,
                reward=0.0,
                done=True,
            )
        
        current_email = self.emails[self.current_idx]
        return EmailState(
            current_email=current_email,
            inbox_size=len(self.emails),
            processed_count=self.current_idx,
            reward=self.episode_reward,
            done=False,
        )

    def step(self, action: Action) -> Tuple[EmailState, float, bool]:
        if self.current_idx >= len(self.emails):
            return self._get_state(), 0.0, True

        current_email = self.emails[self.current_idx]
        
        # Grade the action
        reward = self.grader.grade_action(
            action.action_type,
            current_email.is_spam,
            current_email.is_important,
        )
        
        self.episode_reward += reward
        self.actions_taken.append(action)
        self.current_idx += 1
        
        next_state = self._get_state()
        done = next_state.done
        
        return next_state, reward, done

    def get_metrics(self) -> EpisodeMetrics:
        """Get episode metrics."""
        if not self.emails:
            return EpisodeMetrics(
                total_reward=0.0, emails_processed=0, accuracy=0.0,
                precision=0.0, recall=0.0
            )
        
        metrics = self.grader.compute_metrics(
            [a.action_type for a in self.actions_taken],
            [(e.is_spam, e.is_important) for e in self.emails[:self.current_idx]],
        )
        
        return EpisodeMetrics(
            total_reward=self.episode_reward,
            emails_processed=self.current_idx,
            accuracy=metrics.get("accuracy", 0.0),
            precision=metrics.get("accuracy", 0.0),  
            recall=metrics.get("accuracy", 0.0) 
        )

    def get_task_description(self) -> str:
        
        return self.task.get_description()
    def state(self):
        return self._get_state()