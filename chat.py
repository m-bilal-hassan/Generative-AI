
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(
    "openai/gpt-oss-20b",
    model_provider="groq"
)

print("🤖 AI Chatbot Started!")
print("Type 'exit' to stop.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Chatbot closed. Goodbye!")
        break

    try:
        response = model.invoke(user_input)
        print("AI:", response.content)
    except Exception as e:
        print("Error:", e)
