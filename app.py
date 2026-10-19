
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Load API key from .env
load_dotenv()

# Page settings
st.set_page_config(
    page_title="My AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 My AI Chatbot")
st.caption("Ask anything and get simple, short answers.")

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    st.write("Powered by LangChain + Groq")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Load AI model
@st.cache_resource
def get_model():
    return init_chat_model(
        "openai/gpt-oss-20b",
        model_provider="groq",
        temperature=0.3
    )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Ask me anything...")

if prompt:
    # Display user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                model = get_model()

                # Convert chat history into LangChain messages
                chat_history = [
                    SystemMessage(
                        content=(
                            "You are a helpful AI assistant. "
                            "Give short, simple and direct answers "
                            "in easy English. For normal questions, "
                            "answer in 2-4 sentences maximum. "
                            "Avoid unnecessary details and long lists. "
                            "Only provide detailed explanations when "
                            "the user explicitly asks for details. "
                            "For coding questions, provide the necessary "
                            "code with a brief explanation."
                        )
                    )
                ]

                for message in st.session_state.messages:
                    if message["role"] == "user":
                        chat_history.append(
                            HumanMessage(content=message["content"])
                        )
                    elif message["role"] == "assistant":
                        chat_history.append(
                            AIMessage(content=message["content"])
                        )

                response = model.invoke(chat_history)
                answer = response.content

                st.markdown(answer)

                # Save assistant response
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as e:
                st.error(f"Something went wrong: {e}")
