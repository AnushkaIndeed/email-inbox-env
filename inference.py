import os
from pathlib import Path
from env.email_env import EmailEnvironment
from env.models import Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN", os.getenv("API_KEY", None)) # Try both names

# OpenAI client initialization
client = None
if HF_TOKEN:
    from openai import OpenAI
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN
    )

def decide_action_with_llm(email_text: str):
    """
    Decide action using the LiteLLM proxy. 
    This is required to pass the LLM Criteria Check.
    """
    # Baseline action if no client is available
    fallback_action = "classify"
    if "win" in email_text.lower() or "offer" in email_text.lower():
        fallback_action = "delete"

    if not client:
        return fallback_action

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system", 
                    "content": "You are an intelligent email inbox assistant. Analyze the email and choose the most appropriate action types from: [classify, delete, archive, move]. Respond with ONLY the action name."
                },
                {"role": "user", "content": f"Decide action for this email: {email_text[:500]}"} # Truncate for efficiency
            ],
            max_tokens=10,
            temperature=0
        )
        action = response.choices[0].message.content.strip().lower()
        
        # Validate that the action is one of the supported types
        valid_actions = ["classify", "delete", "archive", "move"]
        if action in valid_actions:
            return action
        return fallback_action
        
    except Exception as e:
        print(f"LLM Call failed: {e}")
        return fallback_action


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
            email_text = f"Subject: {state.current_email.subject}\nBody: {state.current_email.body}"
        else:
            email_text = "no email"

        # Decide action via LLM (to satisfy validator criteria)
        action_type = decide_action_with_llm(email_text)

        # Create action
        action = Action(
            action_type=action_type,
            confidence=0.8
        )

        try:
            result = env.step(action)
            # Support both (state, reward, done, info) and (state, reward, done)
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
            next_state = state 

        rewards.append(reward)

        # STEP FORMAT (Required by validator)
        print(
            f"[STEP] step={step} action={action_type} "
            f"reward={reward:.2f} done={str(done).lower()} "
            f"error={error if error else 'null'}"
        )

        state = next_state

    # Success condition
    success = True if sum(rewards) > 0 else False

    # END FORMAT (Required by validator)
    print(
        f"[END] success={str(success).lower()} steps={step} "
        f"rewards={','.join(f'{r:.2f}' for r in rewards)}"
    )

if __name__ == "__main__":
    # Run multiple tasks to demonstrate multi-task capability
    for task in ["spam", "important", "organize"]:
        run_inference(task)