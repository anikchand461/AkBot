import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chat_models import init_chat_model
from langchain.chains import RetrievalQA
from langchain.retrievers import EnsembleRetriever  

load_dotenv()

# 1. Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 2. Create 4 DBs from different txt files
db1 = Chroma.from_documents(TextLoader("knowledge_base/basic_details.txt").load(), embeddings, persist_directory="./chroma_db1")
db2 = Chroma.from_documents(TextLoader("knowledge_base/blogs.txt").load(), embeddings, persist_directory="./chroma_db2")
db3 = Chroma.from_documents(TextLoader("knowledge_base/linkedin.txt").load(), embeddings, persist_directory="./chroma_db3")
db4 = Chroma.from_documents(TextLoader("knowledge_base/profiles.txt").load(), embeddings, persist_directory="./chroma_db4")

# 3. Create retrievers for each
retriever1 = db1.as_retriever()
retriever2 = db2.as_retriever()
retriever3 = db3.as_retriever()
retriever4 = db4.as_retriever()

# 4. Merge retrievers into one
merged_retriever = EnsembleRetriever(
    retrievers=[retriever1, retriever2, retriever3, retriever4],
    weights=[0.25, 0.25, 0.25, 0.25]   
)

# 5. Gemini Flash model
llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# 6. Retrieval-Augmented QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=merged_retriever,
    return_source_documents=True
)

# 7. Chat Example
query = "What is Anik's resume link?"
result = qa_chain({"query": query})

print("Answer:", result["result"])
