# 💬 AI Chatbot  

An AI-powered **personalized chatbot** built using **RAG + Gemini API**, integrated with **FastAPI** backend, **ChromaDB** for vector storage, and a responsive **JS frontend**. Deployed seamlessly on **Render**.  

---

## 🚀 Features  
- 🧠 **Retrieval-Augmented Generation (RAG)** for contextual answers from personal/project data  
- ⚡ **FastAPI Backend** for handling chatbot requests  
- 📂 **ChromaDB** as vector database for efficient embeddings & retrieval  
- 🤖 **Gemini API** integration for LLM responses  
- 🌐 **Frontend** with HTML/CSS/JS for chat interface  
- ☁️ **Deployed on Render** with environment-based API key management  

---

## 🛠 Tech Stack  
- **Backend:** FastAPI, LangChain  
- **Vector DB:** ChromaDB  
- **LLM API:** Gemini API  
- **Frontend:** HTML, CSS, JavaScript  
- **Deployment:** Render  

---

## ⚙️ Setup & Installation  

1. **Clone the repo**  
   ```bash
   git clone https://github.com/anikchand461/AkBot.git
   cd ai-chatbot
   ```

2. **Create virtual environment & install dependencies**  
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up environment variables**  
   Create a `.env` file in the project root:  
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

4. **Run the backend**  
   ```bash
   uvicorn app.main:app --reload
   ```

5. **Open the frontend**  
   Open `index.html` in your browser, or serve via any static hosting.  

---

## 📦 Deployment on Render  

1. Push your repo to GitHub.  
2. Create a **Render Web Service** → select FastAPI backend.  
3. Add `GEMINI_API_KEY` under **Environment Variables** in Render Dashboard.  
4. Deploy 🚀  

---