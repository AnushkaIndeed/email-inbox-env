import os
from openai import OpenAI
from env.email_env import EmailEnvironment
from env.models import Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")


client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

def decide_action_with_llm(email_text: str):
    if "win" in email_text.lower() or "offer" in email_text.lower():
        return "delete"
    return "classify"


def run_inference(task_type="spam"):
    env = EmailEnvironment(task_type=task_type)

    state = env.reset()

    step = 0
    rewards = []
    done = False

    # START LINE
    print(f"[START] task={task_type} env=email_env model={MODEL_NAME}")

    while not done:
        step += 1

        # Get email text safely
        if state.current_email:
            email_text = f"{state.current_email.subject} {state.current_email.body}"
        else:
            email_text = "no email"

        # Decide action 
        action_type = decide_action_with_llm(email_text)

        # Create action
        action = Action(
            action_type=action_type,
            confidence=0.8
        )

        try:
            result = env.step(action)

            if len(result) == 4:
                next_state, reward, done, info = result
                error = info.get("error", None)
            else:
                next_state, reward, done = result
                error = None

        except Exception as e:
            reward = 0.0
            done = True
            error = str(e)
            next_state = state  # prevent crash

        rewards.append(reward)

        # STEP FORMAT
        print(
            f"[STEP] step={step} action={action_type} "
            f"reward={reward:.2f} done={str(done).lower()} "
            f"error={error if error else 'null'}"
        )

        state = next_state

    #  success condition
    success = True if sum(rewards) > 0 else False

    #  END FORMAT
    print(
        f"[END] success={str(success).lower()} steps={step} "
        f"rewards={','.join(f'{r:.2f}' for r in rewards)}"
    )

if __name__ == "__main__":
    for task in ["spam", "important", "organize"]:
        run_inference(task)