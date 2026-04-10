"""Pydantic models for email inbox environment."""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from typing import Optional


class Email(BaseModel):
    """Email data model."""
    id: str
    sender: str
    subject: str
    body: str
    timestamp: datetime
    is_spam: bool = False
    is_important: bool = False
    has_attachment: bool = False

class EmailState(BaseModel):
    current_email: Optional[Email] = None
    done: bool = False
    processed_count: int = 0


class Action(BaseModel):
    """Action model for agent."""
    action_type: str = Field(..., description="classify, archive, delete, or move")
    target_folder: Optional[str] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class EpisodeMetrics(BaseModel):
    """Metrics for an episode."""
    total_reward: float
    emails_processed: int
    accuracy: float
    precision: float
    recall: float
