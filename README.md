---
title: IntelliMail RL Environment
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# IntelliMail — Email Inbox RL Environment

An RL environment that simulates an email inbox. The agent has to learn how to manage emails — detecting spam, flagging important ones, and keeping things organized. Built for the Meta PyTorch hackathon using the OpenEnv framework.

## What it does

The environment gives the agent one email at a time and the agent picks an action (`classify`, `delete`, `archive`, or `move`). After processing all emails, we evaluate how well it did based on the task.

There are three tasks:
- **Spam Detection** — correctly identify and delete spam while keeping legit emails
- **Important Email** — flag important emails so the user sees them first
- **Inbox Organization** — sort everything into the right place

Each task has its own grading logic and scoring.

## Project structure

```
email-inbox/
├── inference.py          # main script — runs the agent through all tasks
├── server/
│   └── app.py            # FastAPI server (reset/step endpoints)
├── env/
│   ├── email_env.py      # core RL environment (step, reset, metrics)
│   ├── tasks.py          # task definitions + grading
│   ├── models.py         # pydantic models (Email, Action, State, etc.)
│   └── grader.py         # metric computation
├── data/
│   └── emails.json       # dataset of 10 sample emails
├── openenv.yaml          # OpenEnv config
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── dashboard.html        # web UI for the inbox
└── login.html            # login page
```

## How to run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run inference locally

```bash
python inference.py
```

This runs all three tasks (spam, important, organize) sequentially and prints step-by-step logs like:

```
[START] task=spam env=email_env model=gpt-4.1-mini
[STEP] step=1 action=delete reward=0.85 done=false error=null
[STEP] step=2 action=classify reward=0.50 done=false error=null
...
[END] success=true steps=10 score=0.68 rewards=0.85,0.50,...
```

### Run the server

```bash
python -m server.app
```

This starts a FastAPI server on port 7860 with:
- `POST /reset` — reset the env for a task
- `POST /step` — take an action
- `GET /` — login page
- `GET /dashboard` — interactive dashboard

### Docker

```bash
docker build -t email-inbox-env .
docker run -p 7860:7860 email-inbox-env
```

## How the environment works

```python
from env.email_env import EmailEnvironment
from env.models import Action

env = EmailEnvironment(task_type="spam")
state = env.reset()

while not state.done:
    action = Action(action_type="delete")  # or classify/archive/move
    state, reward, done = env.step(action)
    print(f"reward: {reward}, done: {done}")

metrics = env.get_metrics()
print(f"accuracy: {metrics.accuracy}")
```

Each step returns:
- `state` — next email to process + current score
- `reward` — how good the action was for this specific email
- `done` — whether all emails have been processed

## LLM integration

If `HF_TOKEN` (or `API_KEY`) is set, the agent uses an LLM to decide actions. Otherwise it falls back to simple keyword heuristics (e.g. emails containing "win" or "offer" → delete).

The LLM gets the email text and picks from `[classify, delete, archive, move]`.

## Scoring

All scores are kept strictly between 0 and 1 (exclusive). The final task score is based on accuracy — how many emails the agent handled correctly for that specific task.

| Task | Correct action (spam) | Correct action (important) | Correct action (normal) |
|------|----------------------|---------------------------|------------------------|
| Spam Detection | delete spam, keep rest | — | — |
| Important Email | — | classify important | don't classify normal |
| Organization | delete spam, classify important | classify important | classify/move/archive |

## Email data format

Each email in `data/emails.json`:

```json
{
  "id": "email_001",
  "sender": "offers@spam-store.com",
  "subject": "You've Won a Prize!",
  "body": "Click here to claim...",
  "timestamp": "2024-01-15T09:30:00",
  "is_spam": true,
  "is_important": false,
  "has_attachment": false
}
```

## Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API endpoint | `https://api.openai.com/v1` |
| `MODEL_NAME` | Model to use | `gpt-4.1-mini` |
| `HF_TOKEN` | API key for LLM calls | None (uses heuristic fallback) |

## Tech stack

- Python 3.9+
- FastAPI + Uvicorn
- Pydantic v2
- OpenAI SDK
- OpenEnv framework
- Docker

## License

MIT
