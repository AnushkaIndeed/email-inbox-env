from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from env.email_env import EmailEnvironment
from env.models import Action
import os
import uvicorn

# Define base directory for locating static UI files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

env = EmailEnvironment()
state = env.reset()

app = FastAPI()

@app.get("/")
async def read_login():
    return FileResponse(os.path.join(BASE_DIR, 'login.html'))

@app.get("/dashboard")
async def read_dashboard():
    return FileResponse(os.path.join(BASE_DIR, 'dashboard.html'))

@app.post("/reset")
async def reset_api():
    global state
    state = env.reset()
    
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
        "reward": state.reward,
        "processed_count": state.processed_count,
        "inbox_size": state.inbox_size,
        "done": state.done
    }

@app.post("/step")
async def step_api(request: Request):
    global state
    data = await request.json()
    action_type = data.get("action")
    
    act = Action(action_type=action_type, confidence=1.0)
    next_state, reward, done = env.step(act)
    state = next_state

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
        "reward": reward,
        "total_reward": state.reward,
        "processed_count": state.processed_count,
        "inbox_size": state.inbox_size,
        "done": done,
        "task_score": state.score
    }

def main():
    """Entry point for the server script."""
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
