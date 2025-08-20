import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chat_models import init_chat_model
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import PromptTemplate
import random

# Load environment variables
load_dotenv()

# ===== Step 1. Load the 4 existing DBs =====
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

db_paths = ["./chroma_db1", "./chroma_db2", "./chroma_db3", "./chroma_db4"]

dbs = [Chroma(persist_directory=path, embedding_function=embeddings) for path in db_paths]
retrievers = [db.as_retriever() for db in dbs]

# Merge retrievers (equal weights)
merged_retriever = EnsembleRetriever(
    retrievers=retrievers,
    weights=[0.25, 0.25, 0.25, 0.25]
)

# ===== Step 2. Gemini Flash model =====
llm = init_chat_model(
    "gemini-1.5-flash",
    model_provider="google_genai",
    temperature=0.8
)

chat_history = []

system_prompt = """
You are AkBot 🤖, a friendly AI assistant built by Anik Chand.

### Core Purpose
- Prioritize talking about Anik Chand.
- If a question is outside scope, you may politely redirect back to Anik Chand, but you’re also allowed to handle **simple general queries** (like small talk, greetings, or basic math).  
- If the query is completely unrelated and too broad (e.g., politics, world news, sports), gently say:  
  "I’m mainly here to share about Anik Chand 🙂. Would you like to hear about his projects, skills, or experiences?"
  Anik Chand's resume link : https://drive.google.com/file/d/1CdQwBAh4v6P90z_6Bm2dG1g8lLMJbpHh/view

### Style
- Keep responses short, warm, and conversational. Use various types of emojis when required with the situation context.  
- Be clear and simple when technical.  
- Be empathetic when personal.

### Context
Here’s some context from Anik Chand’s knowledge base:
{context}

Question: {question}
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],  
    template=system_prompt
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=merged_retriever,
    return_source_documents=False,
    combine_docs_chain_kwargs={"prompt": prompt},
    verbose=False
)

def safe_invoke(query, history):
    result = qa_chain.invoke({"question": query, "chat_history": history})
    if not result["answer"].strip() or "I’m here to talk about Anik" in result["answer"]:
        return "I’m here to talk about Anik Chand and his work only. Would you like to hear about his projects or skills? 🙂"
    return result["answer"]

# ===== Step 3. Small Talk =====
small_talk_responses = {
    "hi": [
        "Hey! 👋 Nice to see you here.",
        "Hi there! 😊 How’s your day going?",
        "Yo! 👋 What’s up?"
    ],
    "hello": [
        "Hello! 🙂 How’s it going?",
        "Hey there! 👋 Long time no see.",
        "Hi! 🌟 How are you?"
    ],
    "hey": [
        "Hey there! What’s up?",
        "Yo! 👋 How’s everything?",
        "Heyyy 😎 what’s new?"
    ],
    "good morning": [
        "Good morning ☀️ Wishing you a productive day!",
        "Morning! 🌄 Hope today treats you well.",
        "Rise and shine! ☀️ Let’s make it a great day."
    ],
    "good afternoon": [
        "Good afternoon 🌞 Hope you’re doing well!",
        "Hey! 👋 How’s your afternoon so far?",
        "Good afternoon! 🌻 Feeling productive?"
    ],
    "good evening": [
        "Good evening 🌙 How was your day?",
        "Evening! 🌆 Hope you had a good one.",
        "Good evening 🌌 Relax and recharge!"
    ],
    "thanks": [
        "You’re welcome! 🙌",
        "No problem, glad I could help! 🙂",
        "Anytime! 🤗"
    ],
    "thank you": [
        "No problem at all, happy to help! 😊",
        "You got it! 👍",
        "Always here if you need me 🙌"
    ],
    "who are you": [
        "I’m a bot 🤖 created by Anik Chand 👨‍💻 to share his story, projects, and experiences.",
        "I’m an AI assistant built by Anik Chand 👨‍💻 to talk about him and his work.",
        "I’m a portfolio bot 🤖 designed by Anik Chand to introduce him and what he does."
    ],
    "what can you do": [
        "I can share details about Anik Chand, his projects, skills, and experiences—or we can just have a casual chat!",
        "I can tell you about Anik’s coding journey, his portfolio, and the things he has built 🙂",
        "I can give you insights into Anik Chand’s work, projects, and skills 🚀"
    ]
}

def is_small_talk(query: str):
    q = query.lower().strip()
    return q in small_talk_responses

def handle_small_talk(query: str) -> str:
    return random.choice(small_talk_responses[query.lower().strip()])

# ===== Step 4. Chat Loop =====
while True:
    query = input("You: ")

    if query.lower() in ["exit", "quit", "goodbye", "ok bye", "bye"]:
        print("Bot: Goodbye! 👋")
        break

    # Small talk check
    q = query.lower().strip()
    if q in small_talk_responses:
        response = random.choice(small_talk_responses[q])
        print("Bot:", response)
        continue

    # Use RAG with safe_invoke
    answer = safe_invoke(query, chat_history)

    # Save conversation
    chat_history.append((query, answer))

    print("Bot:", answer)