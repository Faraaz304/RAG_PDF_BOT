import gradio as gr
import os
import time
# Import all the backend functions from our RAG script
from rag_backend import (
    load_document,
    split_text,
    create_embeddings,
    create_vectorstore,
    create_llm,
    setup_qa_chain
)

# --- Global State Management ---
# This variable will hold the initialized QA chain so it's not re-created on every call.
qa_chain_instance = None

# --- Core Logic Functions for the Gradio App ---


def process_document_and_setup_chain(file):
    """
    This function is triggered when a file is uploaded.
    It runs the full RAG pipeline and prepares the QA chain.
    This version includes more frequent updates to prevent timeouts.
    """
    global qa_chain_instance
    
    if file is None:
        return "Please upload a document to begin.", []

    file_path = file
    
    try:
        # --- Stage 1: Loading ---
        yield "🔄 Stage 1/4: Loading document...", []
        documents = load_document(file_path)
        
        # --- Stage 2: Splitting ---
        yield f"🔄 Stage 2/4: Splitting {len(documents)} pages into chunks...", []
        chunks = split_text(documents)
        
        # --- Stage 3: Embedding and Vector Store Creation (The Slow Part) ---
        yield f"🔄 Stage 3/4: Creating embeddings for {len(chunks)} chunks... (This may take a while)", []
        embeddings = create_embeddings()
        vectorstore = create_vectorstore(chunks, embeddings)
        
        # --- Stage 4: Setting up the QA Chain ---
        yield "🔄 Stage 4/4: Setting up the QA system...", []
        llm = create_llm()
        qa_chain_instance = setup_qa_chain(vectorstore, llm)
        
        print("--- ✅ Document processing complete! ---")
        
        # --- Final Success Message ---
        yield "✅ Document processed successfully! You can now ask questions.", []

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        qa_chain_instance = None # Reset the chain on error
        # Send a detailed error message back to the UI
        yield f"An error occurred: {e}", []

def handle_user_interaction(user_input, chat_history):
    """
    This function is triggered when the user sends a message.
    It takes the user's input and the existing chat history.
    """
    global qa_chain_instance
    
    if qa_chain_instance is None:
        # Handle case where user tries to chat before processing a document
        chat_history.append((user_input, "Error: Please upload and process a document first."))
        return "", chat_history

    # Append the user's message to the chat history immediately
    chat_history.append((user_input, None))
    # We yield the updated history to show the user's message on the screen right away
    yield "", chat_history

    # Get the response from the QA chain
    try:
        result = qa_chain_instance.invoke({"query": user_input})
        bot_response = result["result"]
    except Exception as e:
        bot_response = f"An error occurred while getting the answer: {e}"

    # Stream the bot's response to the UI
    response_stream = ""
    for char in bot_response:
        response_stream += char
        time.sleep(0.01) # A small delay creates a nice typing effect
        # Update the last message in the chat history with the streamed response
        chat_history[-1] = (user_input, response_stream)
        yield "", chat_history


# --- Building the Gradio Interface ---

with gr.Blocks(theme=gr.themes.Default(), title="Chat with Your Document") as demo:
    gr.Markdown(
        """
        # 📄 Chat with Your Document using Llama 3.2
        Upload a PDF or TXT file, wait for it to be processed, and then ask questions about its content.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # --- Left Column: File Upload and Status ---
            file_uploader = gr.File(
                label="Upload a .pdf or .txt document",
                file_types=[".pdf", ".txt"],
                type="filepath" # We need the path of the file
            )
            
            status_display = gr.Markdown("Status: Please upload a document to start.")

        with gr.Column(scale=2):
            # --- Right Column: Chat Interface ---
            chatbot_display = gr.Chatbot(
                label="Conversation",
                bubble_full_width=False,
                height=550
            )
            
            message_textbox = gr.Textbox(
                label="Your Question",
                placeholder="Ask a question about your document..."
            )

    # --- Event Handling ---
    
    # Event 1: When a file is uploaded, trigger the processing function.
    file_uploader.upload(
        fn=process_document_and_setup_chain,
        inputs=[file_uploader],
        outputs=[status_display, chatbot_display],
        show_progress="full"
    )

    # Event 2: When the user submits a message, trigger the chat handler function.
    message_textbox.submit(
        fn=handle_user_interaction,
        inputs=[message_textbox, chatbot_display],
        outputs=[message_textbox, chatbot_display] # Clears the textbox and updates the chat
    )

# --- Launch the App ---
if __name__ == "__main__":
    # share=True creates a public link for 72 hours.
    # debug=True provides more detailed error messages in the console.
    demo.launch(debug=True, share=False)