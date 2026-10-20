
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

model_id = "Athuin/tinyLama-german"

pipe = pipeline(
    "text-generation",
    model=model_id,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.3,
    return_full_text=False
)

llm = HuggingFacePipeline(pipeline=pipe)

print("🤖 Local LLM Started!")
print("Type 'exit' to stop.\n")

while True:
    question = input("You: ").strip()

    if question.lower() == "exit":
        print("Chatbot closed!")
        break

    if not question:
        continue

    prompt = (
        "Answer the following question clearly and accurately "
        "in simple English.\n"
        f"Question: {question}\n"
        "Answer:"
    )

    try:
        response = llm.invoke(prompt)
        print("AI:", response.strip(), "\n")
    except Exception as e:
        print("Error:", e)
