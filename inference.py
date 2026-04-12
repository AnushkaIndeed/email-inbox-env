import os
from env.email_env import EmailEnvironment
from env.models import Action

# ---------------- CONFIG ----------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN", os.getenv("API_KEY", None))

# ---------------- OPENAI CLIENT ----------------
client = None
if HF_TOKEN:
    from openai import OpenAI
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN
    )


# ---------------- SAFE SCORE ----------------
def safe_score(x: float) -> float:
    """Ensure score is strictly within (0,1)"""
    return max(0.1, min(0.9, float(x)))


# ---------------- ACTION DECISION ----------------
def decide_action_with_llm(email_text: str) -> str:
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
                    "content": "Choose one action from: classify, delete, archive, move. Respond with only the action."
                },
                {"role": "user", "content": email_text[:500]}
            ],
            max_tokens=10,
            temperature=0
        )

        action = response.choices[0].message.content.strip().lower()
        if action in ["classify", "delete", "archive", "move"]:
            return action

        return fallback_action

    except Exception as e:
        print(f"[DEBUG] LLM failed: {e}")
        return fallback_action


# ---------------- INFERENCE ----------------
def run_inference(task_type: str):
    env = EmailEnvironment(task_type=task_type)
    state = env.reset()

    step = 0
    rewards = []
    done = False

    print(f"[START] task={task_type} env=email_env model={MODEL_NAME}")

    while not done:
        step += 1

        if state.current_email:
            email_text = f"{state.current_email.subject} {state.current_email.body}"
        else:
            email_text = "empty email"

        action_type = decide_action_with_llm(email_text)

        action = Action(
            action_type=action_type,
            confidence=0.5
        )

        try:
            next_state, reward, done = env.step(action)
            error = None
        except Exception as e:
            reward = 0.5
            done = True
            error = str(e)
            next_state = state

        reward = safe_score(reward)
        rewards.append(reward)

        print(
            f"[STEP] step={step} action={action_type} "
            f"reward={reward:.2f} done={str(done).lower()} "
            f"error={error if error else 'null'}"
        )

        state = next_state

    metrics = env.get_metrics()
    score = safe_score(metrics.accuracy)

    success = score > 0.3

    print(
        f"[END] success={str(success).lower()} steps={step} "
        f"score={score:.2f} "
        f"rewards={','.join(f'{r:.2f}' for r in rewards)}"
    )

    return score


# ---------------- MAIN ----------------
if __name__ == "__main__":
    for task in ["spam", "important", "organize"]:
        run_inference(task)