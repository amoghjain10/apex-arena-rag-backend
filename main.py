import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from pinecone import Pinecone # Import Pinecone client
from langchain_pinecone import PineconeVectorStore # Import PineconeVectorStore

# Load environment variables
load_dotenv()

# Get API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = "apex-arena-rag" # Match the index name in ingest.py

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env file.")
if not PINECONE_ENVIRONMENT:
    raise ValueError("PINECONE_ENVIRONMENT not found in .env file.")

# --- FastAPI App Setup ---
app = FastAPI(
    title="Apex Arena RAG Chatbot Backend",
    description="Backend API for The Apex Arena Chatbot using Gemini and RAG."
)

# --- CORS Configuration ---
origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "https://yourapexarena.framer.website", # Replace with your actual Framer site URL
    "https://*.framer.website",
    "https://*.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- RAG Components Initialization ---
# Initialize LLM (Gemini model)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", google_api_key=GOOGLE_API_KEY, temperature=0.7)

# Initialize embeddings for RAG
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)

# Initialize Pinecone and connect to the index
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings) # The 'embedding' arg handles passing GoogleGenerativeAIEmbeddings

# Create retriever
retriever = vectorstore.as_retriever() # Default search_kwargs={"k": 4}

# Define RAG prompt template
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
document_chain = create_stuff_documents_chain(llm, RAG_PROMPT)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# --- API Request/Response Models ---
class ChatRequest(BaseModel):
    user_message: str

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
        response = retrieval_chain.invoke({"input": user_input})
        bot_answer = response.get("answer", "I apologize, I couldn't process that. Please try again.")
        return ChatResponse(bot_message=bot_answer)

    except Exception as e:
        print(f"Error in chat_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")