from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from env.email_env import EmailEnvironment
from env.models import Action
import os
import uvicorn

# Define base directory for locating static UI files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Create one environment per task type as defined in openenv.yaml
envs = {
    "spam": EmailEnvironment(task_type="spam"),
    "important": EmailEnvironment(task_type="important"),
    "organize": EmailEnvironment(task_type="organize"),
}
# Default env for the dashboard
default_env = envs["spam"]
states = {k: v.reset() for k, v in envs.items()}

app = FastAPI()


def safe_score(x: float) -> float:
    """Guarantee score is strictly within (0, 1) — never 0.0 or 1.0."""
    return round(max(0.02, min(0.98, float(x))), 4)


@app.get("/")
async def read_login():
    return FileResponse(os.path.join(BASE_DIR, 'login.html'))


@app.get("/dashboard")
async def read_dashboard():
    return FileResponse(os.path.join(BASE_DIR, 'dashboard.html'))


@app.post("/reset")
async def reset_api(request: Request):
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass

    task_name = body.get("task", "spam")
    if task_name not in envs:
        task_name = "spam"

    env = envs[task_name]
    state = env.reset()
    states[task_name] = state

    email_data = None
    if state.current_email:
        email_data = {
            "id": state.current_email.id,
            "sender": state.current_email.sender,
            "subject": state.current_email.subject,
            "body": state.current_email.body,
            "timestamp": state.current_email.timestamp.isoformat() if state.current_email.timestamp else None,
            "is_important": state.current_email.is_important,
            "has_attachment": state.current_email.has_attachment
        }

    return {
        "observation": email_data,
        "reward": safe_score(state.reward),       # ← clamped
        "score": safe_score(state.score),         # ← clamped
        "processed_count": state.processed_count,
        "inbox_size": state.inbox_size,
        "done": state.done,
        "task": task_name,
    }


@app.post("/step")
async def step_api(request: Request):
    data = await request.json()
    action_type = data.get("action")
    task_name = data.get("task", "spam")
    if task_name not in envs:
        task_name = "spam"

    env = envs[task_name]
    act = Action(action_type=action_type, confidence=0.5)  # neutral confidence
    next_state, reward, done = env.step(act)
    states[task_name] = next_state

    email_data = None
    if next_state.current_email:
        email_data = {
            "id": next_state.current_email.id,
            "sender": next_state.current_email.sender,
            "subject": next_state.current_email.subject,
            "body": next_state.current_email.body,
            "timestamp": next_state.current_email.timestamp.isoformat() if next_state.current_email.timestamp else None,
            "is_important": next_state.current_email.is_important,
            "has_attachment": next_state.current_email.has_attachment
        }

    return {
        "observation": email_data,
        "reward": safe_score(reward),                  # ← was round(reward, 2) — bug fixed
        "score": safe_score(next_state.score),         # ← clamped
        "total_reward": safe_score(next_state.reward), # ← clamped
        "processed_count": next_state.processed_count,
        "inbox_size": next_state.inbox_size,
        "done": done,
        "task": task_name,
    }


def main():
    """Entry point for the server script."""
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()