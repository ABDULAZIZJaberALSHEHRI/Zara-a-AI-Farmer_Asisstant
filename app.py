import gradio as gr

# Import modules
from modules.disease_detector import predict_image, analyze_uploaded_plant_image
from modules.knowledge_base import prepare_chroma_from_local_pdfs
from modules.chat import agent_chatbot_response, clear_chat
from modules.audio import transcribe_audio
from modules.ui import get_custom_css, get_logo_html
from config import OPENAI_API_KEY, BACKGROUND_IMAGE_PATH, LOGO_PATH


def handle_uploaded_plant_image(image_path, chat_history):
    label = analyze_uploaded_plant_image(image_path)
    if label and not label.startswith("⚠️"):
        question = f"How can i grow {label} ?."
        chat_history = agent_chatbot_response(question, chat_history)
    else:
        chat_history.append(("System", label))
    return chat_history

# Build the application UI
def build_app():
    custom_css = get_custom_css()
    logo_html = get_logo_html()

    with gr.Blocks(css=custom_css) as app:
        gr.HTML(f"""
            <div class="app-header">
                {logo_html}
                <p>AI-powered tools to help with plant disease identification and farming knowledge</p>
            </div>
            <div style="position: absolute; top: 10px; left: 15px; color: white; font-size: 14px;">
                © 2025 Abdulaziz Al-Shahri — <a href='https://www.linkedin.com/in/abdulaziz-alshehri-241252182' target='_blank' style="color: #384043; text-decoration: none;">LinkedIn</a>
            </div>
    """)
        with gr.Tabs() as tabs:
            # ==== Tab 1: Plant Disease Detector ====
            with gr.TabItem("🌿 Plant Disease Detector", id=0):
                with gr.Row(elem_id="chat-container"):
                    with gr.Column(scale=1):
                        gr.Markdown("""
                            <div class="info-box">
                                <h3>📋 How to use:</h3>
                                <ol>
                                    <li>Upload a clear image of a plant leaf</li>
                                    <li>Click "Analyze Plant" to detect diseases</li>
                                    <li>View the diagnosis results and treatment recommendations</li>
                                </ol>
                            </div>
                        """)
                        image_input = gr.Image(label="Upload leaf image", type="filepath")
                        with gr.Row():
                            predict_button = gr.Button("🔍 Analyze Plant", variant="primary", size="lg")
                            clear_button = gr.Button("🔄 Clear", size="lg")

                    with gr.Column(scale=2):
                        chatbot1 = gr.Chatbot(
                        label=None,
                        show_label=False,
                        show_copy_button=True,
                        bubble_full_width=False,
                        height=450,
                        elem_classes="gr-chatbot"
                        )
                        user_input_1 = gr.Textbox(
                            placeholder="Ask about farming practices, plant care, crop management...",
                            lines=2,
                            show_label=False,
                            scale=4
                        )
                        send_button_1 = gr.Button("Send 📨", variant="primary", size="lg", scale=1)
                        disease_output = gr.Markdown(label="Disease Prediction")

                        with gr.Accordion("Detailed Results", open=False):
                            top_predictions = gr.Plot(label="Confidence Chart")

                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("<h3>📝 Disease Description</h3>")
                                matched_description = gr.Markdown()
                            with gr.Column():
                                gr.Markdown("<h3>💊 Treatment Recommendations</h3>")
                                treatment_recommendations = gr.Markdown()

            # ==== Tab 2: Farming Assistant Chat ====
            with gr.TabItem("🧑‍🌾 Farming Assistant Chat", id=1):
                with gr.Column(scale=3, elem_id="chat-container"):
                    gr.Markdown("""<div class="info-box"><h4>💬 In this tab you can ask the Zara'a farmer assistant general questions about plant and the ways of grow it by using Voice record,plant image and text input and here Sample Questions</h4>
                        <ul>
                            <li>How do I treat tomato leaf blight?    |    How can I improve soil fertility naturally?    |    What's the best time to plant wheat?</li>
                        </ul></div>
                    """)
                    chatbot2 = gr.Chatbot(
                        label=None,
                        show_label=False,
                        show_copy_button=True,
                        bubble_full_width=False,
                        height=450,
                        elem_classes="gr-chatbot"
                    )
                    with gr.Row():
                        upload_audio = gr.Audio(label="🎤 Voice Input", type="filepath", scale=2)
                        plant_image_input = gr.Image(label="🖼️ Upload plant | fruit | vegetable image", type="filepath", scale=2)
                        user_input_2 = gr.Textbox(
                            placeholder="Ask about farming practices, plant care, crop management...",
                            lines=2,
                            show_label=False,
                            scale=4
                        )
                        with gr.Row():
                            send_button_2 = gr.Button("Send 📨", variant="primary", size="lg", scale=1)
                    with gr.Row():
                        chat_clear_button = gr.Button("Clear Chat 🧹", scale=1)

        # ==== Custom Logic: Analyze image & Ask Chat ====
        def analyze_and_ask(image, chat_history):
            prediction_text, top_preds, description, treatment = predict_image(image)

            if "**Prediction:" in prediction_text:
                disease_name = prediction_text.split("**Prediction:")[1].split("**")[0].strip()
                auto_question = f"give me description about this disease: {disease_name}"
                chat_history = agent_chatbot_response(auto_question, chat_history)  # يرسل لشات مرض النبتة
            else:
                chat_history.append(("System", "⚠️ Disease name could not be extracted."))

            return prediction_text, top_preds, description, treatment, chat_history

        # ==== Button Actions ====

        # ================= Chatbot plant disease prediction ======================
        predict_button.click(
            analyze_and_ask,
            inputs=[image_input, chatbot2],
            outputs=[disease_output, top_predictions, matched_description, treatment_recommendations, chatbot1]
        )

        clear_button.click(
            lambda: (None, "", "", "", "", []),
            inputs=None,
            outputs=[disease_output, top_predictions, matched_description, treatment_recommendations, chatbot1]
        )


        send_button_1.click(
            agent_chatbot_response,
            inputs=[user_input_1, chatbot1],
            outputs=chatbot1
        ).then(
            lambda: "",
            inputs=None,
            outputs=user_input_1
        )

        user_input_1.submit(
            agent_chatbot_response,
            inputs=[user_input_1, chatbot1],
            outputs=chatbot1
        ).then(
            lambda: "",
            inputs=None,
            outputs=user_input_1
        )

        # ================= Chatbot farmer assistant ======================
        send_button_2.click(
            agent_chatbot_response,
            inputs=[user_input_2, chatbot2],
            outputs=chatbot2
        ).then(
            lambda: "",
            inputs=None,
            outputs=user_input_2
        )

        user_input_2.submit(
            agent_chatbot_response,
            inputs=[user_input_2, chatbot2],
            outputs=chatbot2
        ).then(
            lambda: "",
            inputs=None,
            outputs=user_input_2
        )
        upload_audio.change(
            transcribe_audio,
            inputs=upload_audio,
            outputs=user_input_2
        )

        chat_clear_button.click(
            clear_chat,
            outputs=chatbot2
        )

        plant_image_input.change(
            handle_uploaded_plant_image,
            inputs=[plant_image_input, chatbot2],
            outputs=chatbot2
        )
    return app
