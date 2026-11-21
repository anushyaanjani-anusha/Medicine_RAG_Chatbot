import streamlit as st
from rag_engine import load_base, ask_rag

st.set_page_config(
    page_title="Medicine Information Chatbot",
    page_icon="💊",
    layout="centered"
)

st.title("💊 Medicine Information Chatbot")

st.caption("AI-Powered Medicine Information Assistant")

st.write(
    "Ask questions about medicine uses, "
    "common side effects and precautions."
)

my_key = st.secrets["GROQ_API_KEY"]


@st.cache_resource
def initiate_rag():
    return load_base(".")

try:
    index, embed_model, chunks = initiate_rag()
    st.success("Medicine database loaded successfully!")

except Exception as e:
    st.error(f"Error loading medicine database: {e}")
    st.stop()

if "chat_hist" not in st.session_state:
    st.session_state.chat_hist = []

for qa in st.session_state.chat_hist:
    with st.chat_message(qa["role"]):
        st.write(qa["content"])

question = st.chat_input(
    "Ask anything about medicines..."
)

if question:

    st.session_state.chat_hist.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching medicine information..."):

            try:
                answer = ask_rag(
                    question,
                    embed_model,
                    chunks,
                    index,
                    api_key=my_key
                )

                st.write(answer)

                st.session_state.chat_hist.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:
                st.error(f"Error generating answer: {e}")

if st.button("Clear Chat"):
    st.session_state.chat_hist = []
    st.rerun()

st.warning(
    "This chatbot provides educational information only. "
    "It does not replace professional medical advice. "
    "Consult a doctor or pharmacist for personal medicine-related decisions."
)
