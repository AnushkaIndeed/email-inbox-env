import gradio as gr
from fastapi import FastAPI
from env.email_env import EmailEnvironment
from env.models import Action

env = EmailEnvironment()
state = env.reset()

app = FastAPI()

def get_email():
    global state
    
    if state.done or not state.current_email:
        return "No more emails", ""
    
    return state.current_email.subject, state.current_email.body

def take_action(action_type):
    global state
    
    if state.done:
        return "Done!"
    
    action = Action(action_type=action_type, confidence=0.8)
    
    next_state, reward, done = env.step(action)
    state = next_state
    
    return f"Action: {action_type} | Reward: {reward}"

def reset():
    global state
    state = env.reset()
    return get_email()

with gr.Blocks() as demo:
    
    gr.Markdown("# 📧 Email Inbox RL Environment")
    gr.Markdown("Email Classification Agent")

    with gr.Tabs():

        with gr.Tab("📥 Playground"):
            subject = gr.Textbox(label="Email Subject")
            body = gr.Textbox(label="Email Body")

            btn_load = gr.Button("Load Email")
            btn_delete = gr.Button("Delete (Spam)")
            btn_classify = gr.Button("Classify")
            btn_reset = gr.Button("Reset")

            output = gr.Textbox(label="Output")

            btn_load.click(fn=get_email, outputs=[subject, body])
            btn_delete.click(fn=lambda: take_action("delete"), outputs=output)
            btn_classify.click(fn=lambda: take_action("classify"), outputs=output)
            btn_reset.click(fn=reset, outputs=[subject, body])

        with gr.Tab("📊 Logs"):
            gr.Markdown("Check Hugging Face Logs tab for RL inference output.")

        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ### Email Inbox RL Environment
            
            - Tasks: Spam Detection, Important Email, Organization
            - RL-based environment simulation
            - Uses structured inference pipeline
            """)
        demo.load(fn=get_email, outputs=[subject, body])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

app = gr.mount_gradio_app(app, demo, path="/ui")