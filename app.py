import streamlit as st
import logging

from rag import ask_question

# =========================
# LOGGING SETUP
# =========================

logging.basicConfig(
    filename="chatbot.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Internal RAG Chatbot"
)

st.title(
    "Company Chatbot"
)

st.caption(
    "Hybrid Search + Qwen3"
)

# =========================
# SESSION STATE
# =========================

if "messages" not in st.session_state:

    st.session_state.messages = []

# =========================
# DISPLAY CHAT HISTORY
# =========================

for msg in st.session_state.messages:

    with st.chat_message(
        msg["role"]
    ):

        st.write(
            msg["content"]
        )

# =========================
# USER INPUT
# =========================

query = st.chat_input(
    "Ask a question..."
)

if query:

    # =========================
    # SAVE USER MESSAGE
    # =========================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message(
        "user"
    ):

        st.write(
            query
        )

    # =========================
    # BUILD CHAT HISTORY
    # =========================

    history = "\n".join(
        f"{msg['role']}: {msg['content']}"
        for msg in st.session_state.messages[-6:]
    )

    # =========================
    # LOG QUESTION
    # =========================

    logging.info(
        f"QUESTION: {query}"
    )

    # =========================
    # ASK RAG
    # =========================

    try:

        result = ask_question(
            query,
            history
        )

        answer = result["answer"]

        sources = result["sources"]

    except Exception as e:

        answer = (
            "An error occurred while "
            "processing your request.\n\n"
            f"{str(e)}"
        )

        sources = []

        logging.error(
            f"ERROR: {str(e)}"
        )

    # =========================
    # DISPLAY RESPONSE
    # =========================

    with st.chat_message(
        "assistant"
    ):

        st.write(
            answer
        )

        if sources:

            st.markdown(
                "### Sources"
            )

            for src in sources:

                st.write(
                    f"- {src}"
                )

    # =========================
    # LOG ANSWER
    # =========================

    logging.info(
        f"ANSWER: {answer}"
    )

    # =========================
    # SAVE ASSISTANT MESSAGE
    # =========================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )