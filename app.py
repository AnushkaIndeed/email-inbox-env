import gradio as gr

# Dummy email data (for UI demo)
emails = [
    {"subject": "Win a lottery!", "body": "Click to claim prize", "type": "spam"},
    {"subject": "Meeting at 5 PM", "body": "Project discussion", "type": "important"},
    {"subject": "50% OFF Offer", "body": "Limited time deal!", "type": "spam"},
]

current_index = 0

def get_email():
    global current_index
    if current_index >= len(emails):
        return "No more emails", ""
    email = emails[current_index]
    return email["subject"], email["body"]

def take_action(action):
    global current_index
    if current_index >= len(emails):
        return "Done!", ""
    
    email = emails[current_index]
    current_index += 1
    
    return f"Action taken: {action}", f"Next email loaded"

def reset():
    global current_index
    current_index = 0
    return get_email()

with gr.Blocks() as demo:
    
    gr.Markdown("# 📧 Email Inbox RL Environment")
    gr.Markdown("Interactive UI for Email Classification Agent")

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
            
            Built for OpenEnv Hackathon 🚀
            """)

demo.launch(server_name="0.0.0.0", server_port=7860)