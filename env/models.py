"""Pydantic models for email inbox environment."""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class Email(BaseModel):
    """Email data model."""
    id: str
    sender: str
    subject: str
    body: str
    timestamp: Optional[datetime] = None
    is_spam: bool = False
    is_important: bool = False
    has_attachment: bool = False

class EmailState(BaseModel):
    """Environment state with explicit score field and tighter safe ranges."""
    current_email: Optional[Email] = None   
    inbox_size: int = 1  # Non-zero integer
    processed_count: int = 0
    reward: float = 0.2  # Start strictly between 0 and 1
    score: float = 0.2   # Explicit score field for validator
    done: bool = False


class Action(BaseModel):
    """Action model for agent."""
    action_type: str = Field(..., description="classify, archive, delete, or move")
    target_folder: Optional[str] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class EpisodeMetrics(BaseModel):
    """Metrics for an episode with all scores in strictly (0, 1)."""
    total_reward: float
    emails_processed: int
    accuracy: float
    precision: float
    recall: float
