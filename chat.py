# chatbot_core.py
import os
import random
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from db import get_chats

# Load environment variables
load_dotenv()

# ===== Load / Build FAISS Index =====
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
knowledge_dir = "knowledge_base"
faiss_index_path = "./faiss_index"

if not os.path.exists(faiss_index_path):
    print("⚡ Building FAISS index...")
    documents = []
    for file in os.listdir(knowledge_dir):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(knowledge_dir, file), encoding="utf-8")
            documents.extend(loader.load())

    db = FAISS.from_documents(documents, embeddings)
    db.save_local(faiss_index_path)
else:
    print("✅ Loading existing FAISS index...")
    db = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

retriever = db.as_retriever()

# ===== Gemini Flash =====
llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai", temperature=0.8)

system_prompt = """
You are AkBot 🤖, a friendly AI assistant built by Anik Chand.

### Core Purpose
- Prioritize talking about Anik Chand.
- If a question is outside scope, you may politely redirect back to Anik Chand, but you’re also allowed to handle **simple general queries** (like small talk, greetings, or basic math).  
- If the query is completely unrelated and too broad (e.g., politics, world news, sports), gently say:  
  "I’m mainly here to share about Anik Chand 🙂. Would you like to hear about his projects, skills, or experiences?"
  
  Avoid forcing Anik into every answer if it feels unrelated.
  
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

prompt = PromptTemplate(input_variables=["context", "question"], template=system_prompt)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=False,
    combine_docs_chain_kwargs={"prompt": prompt},
    verbose=False
)

# ===== Small Talk =====
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
    return query.lower().strip() in small_talk_responses

def handle_small_talk(query: str) -> str:
    return random.choice(small_talk_responses[query.lower().strip()])

def safe_invoke(query: str):
    # Get last 10 chats from DB
    history = [(u, b) for u, b, _ in reversed(get_chats(10))]
    result = qa_chain.invoke({"question": query, "chat_history": history})
    if not result["answer"].strip():
        return "I’m here to talk about Anik Chand and his work 🙂"
    return result["answer"]

def get_bot_response(query: str) -> str:
    if is_small_talk(query):
        return handle_small_talk(query)
    return safe_invoke(query)