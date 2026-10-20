
from dotenv import load_dotenv
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint
)
import os

load_dotenv()

token = os.getenv("HHUGGINGFACEHUB_API_TOKEN")

if not token:
    raise ValueError("Hugging Face token .env mein nahi mila!")

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation",
    huggingfacehub_api_token=token,
    max_new_tokens=512,
    temperature=0.7,
)

model = ChatHuggingFace(llm=llm)

print("🤖 DeepSeek R1 Chatbot Started!")
print("Type 'exit' to stop.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        print("Chatbot closed. Goodbye!")
        break

    if not user_input:
        continue

    try:
        response = model.invoke(user_input)
        print("AI:", response.content, "\n")

    except Exception as e:
        print("Error:", e)
