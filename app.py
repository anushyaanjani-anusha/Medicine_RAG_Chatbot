import streamlit as st
from medicine_engine import ask_medicine

# Page configuration
st.set_page_config(
    page_title="Medicine Information Chatbot",
    page_icon="💊",
    layout="centered"
)

# Title
st.title("💊 Medicine Information Chatbot")

st.caption("AI-powered Medicine Information Assistant")

st.write(
    "Ask questions about medicine uses, "
    "common side effects and precautions."
)

# API key
my_key = st.secrets["GROQ_API_KEY"]

# Initialize chat history
if "chat_hist" not in st.session_state:
    st.session_state.chat_hist = []

# Display previous messages
for qa in st.session_state.chat_hist:
    with st.chat_message(qa["role"]):
        st.write(qa["content"])

# User question
question = st.chat_input(
    "Ask about a medicine..."
)

if question:

    st.session_state.chat_hist.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):

        with st.spinner("Getting medicine information..."):

            try:
                answer = ask_medicine(
                    question,
                    my_key
                )

                st.write(answer)

                st.session_state.chat_hist.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:
                st.error(f"Error: {e}")

# Clear chat
if st.button("Clear Chat"):
    st.session_state.chat_hist = []
    st.rerun()

# Disclaimer
st.warning(
    "This chatbot provides educational information only. "
    "It is not a substitute for professional medical advice. "
    "Consult a doctor or pharmacist for medicine-related decisions."
)
