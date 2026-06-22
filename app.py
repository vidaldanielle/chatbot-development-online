import streamlit as st          # UI framework for building web chatbot interface
import logging                  # For saving logs (questions, answers, errors)
import time                    # For measuring response time
from rag import ask_question   # Your RAG pipeline (retrieval + rerank + LLM)

# =========================
# LOGGING SETUP
# =========================
# Creates a log file where all questions and answers are stored for debugging
logging.basicConfig(
    filename="chatbot.log",     # log file name
    level=logging.INFO,         # log level (INFO and above will be saved)
    format="%(asctime)s - %(message)s"  # log format with timestamp
)

# =========================
# PAGE CONFIG
# =========================
# Sets Streamlit page title (browser tab name)
st.set_page_config(
    page_title="Internal RAG Chatbot"
)

# Main title shown in UI
st.title("Company Chatbot")

# Subtitle / description under title
st.caption("Hybrid Search + Qwen3")

# =========================
# SESSION STATE
# =========================
# Session state is used to store chat history (persists even after rerun)
if "messages" not in st.session_state:
    st.session_state.messages = []   # initialize empty chat history list

# =========================
# DISPLAY CHAT HISTORY
# =========================
# Loop through stored messages and display them on UI
for msg in st.session_state.messages:

    # Create chat bubble based on role (user or assistant)
    with st.chat_message(msg["role"]):

        # Display message content
        st.write(msg["content"])

# =========================
# USER INPUT
# =========================
# Chat input box (bottom of UI)
query = st.chat_input("Ask a question...")

# If user enters a message, execute RAG pipeline
if query:

    # =========================
    # SAVE USER MESSAGE
    # =========================
    # Store user input in session memory
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    # Display user message immediately in chat UI
    with st.chat_message("user"):
        st.write(query)

    # =========================
    # BUILD CHAT HISTORY
    # =========================
    # Convert last 6 messages into text format for LLM context
    history = "\n".join(
        f"{msg['role']}: {msg['content']}"
        for msg in st.session_state.messages[-6:]
    )

    # =========================
    # LOG QUESTION
    # =========================
    # Save user question into log file for debugging / tracking
    logging.info(
        f"QUESTION: {query} | HISTORY_LEN: {len(st.session_state.messages)}"
    )

    # =========================
    # ASK RAG
    # =========================
    elapsed = 0  # initialize response time

    try:
        # Show loading spinner while processing
        with st.spinner("Thinking..."):

            start = time.time()  # start timer

            # Call your RAG pipeline (retrieval + rerank + LLM)
            result = ask_question(query, history)

        elapsed = time.time() - start  # compute response time

        # Extract answer and sources from RAG result
        answer = result["answer"]
        sources = result["sources"]

    except Exception as e:
        # If error occurs, show fallback message
        answer = (
            "An error occurred while "
            "processing your request.\n\n"
            f"{str(e)}"
        )

        sources = []  # empty sources if failure

        # Log error in file
        logging.error(f"ERROR: {str(e)}")

    # =========================
    # DISPLAY RESPONSE
    # =========================
    # Show assistant response in chat UI
    with st.chat_message("assistant"):

        st.write(answer)  # display generated answer

        # Show response time (performance metric)
        st.caption(f"Response Time: {elapsed:.2f}s")

        # If sources exist, display them
        if sources:

            st.markdown(f"### Sources ({len(sources)})")

            for src in sources:
                st.write(f"- {src}")  # list document sources

    # =========================
    # LOG ANSWER
    # =========================
    # Save model output into logs
    logging.info(f"ANSWER: {answer}")

    # =========================
    # SAVE ASSISTANT MESSAGE
    # =========================
    # Store assistant response in session memory for chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })