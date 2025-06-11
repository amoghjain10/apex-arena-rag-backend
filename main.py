import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

# LangChain components for RAG
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file. Please set it.")

# --- FastAPI App Setup ---
app = FastAPI(
    title="Apex Arena RAG Chatbot Backend",
    description="Backend API for The Apex Arena Chatbot using Gemini and RAG."
)

# --- CORS Configuration ---
# Crucial for allowing your Framer frontend to communicate with this backend.
# Replace "*" with your actual Framer website domain(s) in production for security.
origins = [
    "http://localhost:3000", # For local Framer preview
    "http://localhost:8000", # For local backend testing
    "https://theapexarena.framer.website/", # Replace with your actual Framer site URL
    "https://*.framer.website", # Wildcard for Framer's preview/public URLs
    "https://*.vercel.app" # If you deploy other parts on Vercel
    # Add your custom domain here once you set it up in Framer
    # "https://yourcustomdomain.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- RAG Components Initialization ---
CHROMA_DB_DIR = "chroma_db" # Must match directory used in ingest.py

# Initialize embeddings and vector store
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
# Load the persisted ChromaDB
vectordb = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings_model)
retriever = vectordb.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 relevant chunks

# Initialize the Chat model (Gemini Pro)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", google_api_key=GOOGLE_API_KEY, temperature=0.7)

# --- Prompt Template for RAG ---
# This is where you instruct Gemini, provide context, and ask the question.
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
    You are a helpful and friendly customer support bot for The Apex Arena.
    Use the following context to answer the user's question. If the context does not contain the answer,
    state that you cannot answer based on the provided information and then politely
    ask for more clarifying questions to truly understand the user's problem.
    If you still cannot provide a direct answer, instruct the user to email webdeisgnfblawa@gmail.com
    or use the 'Contact Us' section on The Apex Arena website.
    Maintain a polite and professional tone. Do not invent information.

    Relevant context for answering the question:
    {context}
    """),
    ("human", "{input}"),
])

# Create the RAG chain
# Stuff documents chain puts all retrieved documents into the context.
document_chain = create_stuff_documents_chain(llm, RAG_PROMPT)
retrieval_chain = create_retrieval_chain(retriever, document_chain)


# --- API Request/Response Models ---
class Message(BaseModel):
    text: str
    sender: str

class ChatRequest(BaseModel):
    user_message: str
    # Optional: include conversation history if you want Gemini to remember more turns
    # If included, you'd process it before sending to the retrieval_chain
    # chat_history: List[Message] = Field(default_factory=list)

class ChatResponse(BaseModel):
    bot_message: str

# --- Health Check Endpoint ---
@app.get("/")
async def read_root():
    return {"message": "Apex Arena RAG Backend is running!"}

# --- Chat Endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        user_input = request.user_message

        # Invoke the RAG chain
        # The 'input' here is the user's latest message.
        # The 'context' will be retrieved by the retriever part of the chain.
        # The 'system' message is part of RAG_PROMPT
        response = retrieval_chain.invoke({"input": user_input})

        # The response from retrieval_chain contains 'answer' and 'context' (retrieved docs)
        bot_answer = response.get("answer", "I apologize, I couldn't process that. Please try again.")

        return ChatResponse(bot_message=bot_answer)

    except Exception as e:
        print(f"Error in chat_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")