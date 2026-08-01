from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

llm = init_chat_model(
    "gemini-3.5-flash-lite",
    model_provider="google_genai",
    temperature=0.8,
)

response = llm.invoke("Say hello in one sentence.")

print(response.content)