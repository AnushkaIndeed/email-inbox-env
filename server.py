from fastapi import FastAPI
from env.email_env import EmailEnvironment
from env.models import Action

app = FastAPI()

env = EmailEnvironment()
state = env.reset()

@app.post("/reset")
def reset():
    global state
    print("RESET CALLED")  
    state = env.reset()
    return {
        "done": state.done,
        "processed_count": state.processed_count
    }

@app.post("/step")
def step(action: dict):
    global state
    print(f"STEP CALLED with action: {action}") 

    act = Action(action_type=action.get("action"), confidence=0.8)
    next_state, reward, done = env.step(act)
    state = next_state
    
    return {
        "reward": reward,
        "done": done,
        "processed_count": state.processed_count
    }