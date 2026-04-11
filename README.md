---
title: IntelliMail RL Environment
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---
# Email Inbox Environment

A reinforcement learning environment for email inbox management with support for multiple tasks including spam detection, important email prioritization, and inbox organization.

## Project Structure

```
email-inbox-env/
├── env/
│   ├── email_env.py        # Main RL logic
│   ├── models.py           # Pydantic models
│   ├── tasks.py            # 3 task definitions
│   └── grader.py           # Scoring and grading
├── data/
│   └── emails.json         # Test dataset (10 emails)
├── inference.py            # Run inference
├── openenv.yaml            # Environment config
├── Dockerfile              # Docker configuration
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Features

### Tasks
1. **Spam Detection** - Identify and remove spam emails
2. **Important Email** - Prioritize and flag important emails  
3. **Inbox Organization** - Organize emails into appropriate folders

### Components

- **EmailEnvironment** - Main RL environment with step/reset interface
- **Models** - Pydantic models for Email, Action, State, and Metrics
- **Grader** - Scoring system and reward computation
- **Tasks** - Abstract task interface with 3 implementations

## Usage

### Setup

```bash
pip install -r requirements.txt
```

### Run Inference

```bash
# Run with spam detection task (default)
python inference.py

# Run with different tasks
python inference.py important
python inference.py organize
```

### Docker

```bash
# Build image
docker build -t email-inbox-env .

# Run container
docker run email-inbox-env
```

## Data Format

See `data/emails.json` for the email dataset format. Each email contains:
- `id` - Unique email identifier
- `sender` - Sender email address
- `subject` - Email subject
- `body` - Email content
- `timestamp` - Email timestamp
- `is_spam` - Whether email is spam
- `is_important` - Whether email is important
- `has_attachment` - Whether email has attachments

## Configuration

Edit `openenv.yaml` to configure:
- Environment observation and action spaces
- Task parameters
- Training hyperparameters
- Data splits

## API

### EmailEnvironment

```python
from env.email_env import EmailEnvironment
from env.models import Action

# Create environment
env = EmailEnvironment(task_type="spam")

# Reset environment
state = env.reset()

# Take action
action = Action(action_type="delete")
next_state, reward, done = env.step(action)

# Get metrics
metrics = env.get_metrics()
```

### Action Types

- `classify` - Mark email for classification
- `archive` - Archive email
- `delete` - Delete email
- `move` - Move to folder

## Rewards

- **Spam Detection**: +1.0 for correct action, -1.0 for incorrect
- **Important Email**: +1.0 for identifying important, -0.5 for missing
- **Organization**: +0.5 for moving, +0.1 for archiving

## License

MIT
