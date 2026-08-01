from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
import os

load_dotenv()

embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=os.getenv("COHERE_API_KEY")
)