import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# --- Environment Variables ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env file.")
if not PINECONE_ENVIRONMENT:
    raise ValueError("PINECONE_ENVIRONMENT not found in .env file.")

# --- Configuration ---
WEBSITE_URLS = [
    "https://theapexarena.framer.website/",
    "https://theapexarena.framer.website/taylor-swift",
    "https://theapexarena.framer.website/brunch",
    "https://theapexarena.framer.website/trade-show",
    "https://theapexarena.framer.website/basketball",
    "https://theapexarena.framer.website/community-gathering",
    "https://theapexarena.framer.website/seminar",
    "https://theapexarena.framer.website/renting",
    "https://theapexarena.framer.website/about-us",
    "https://theapexarena.framer.website/projects/le-blink",
    "https://theapexarena.framer.website/projects/schlong",
    "https://theapexarena.framer.website/projects/vintage-everything",
    "https://theapexarena.framer.website/projects/senseya",
    "https://theapexarena.framer.website/projects"
]
INDEX_NAME = "apex-arena-rag"
DIMENSIONS = 768 # Confirmed for Google Generative AI Embeddings
METRIC = "cosine"

# --- Data Ingestion Functions ---
def scrape_website(url):
    print(f"Scraping {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract relevant text - adjust selectors based on your website's structure
        paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])
        text_content = ' '.join([para.get_text(separator=' ', strip=True) for para in paragraphs])
        return text_content
    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return None

def save_text_to_file(text, filename):
    os.makedirs("text_data", exist_ok=True)
    filepath = os.path.join("text_data", filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved text to {filepath}")
    return filepath

def load_documents_from_files(text_dir="text_data"):
    documents = []
    for filename in os.listdir(text_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(text_dir, filename)
            loader = TextLoader(filepath)
            documents.extend(loader.load())
    return documents

def ingest_data_to_pinecone():
    print("Starting data ingestion to Pinecone...")

    # 1. Scrape websites and save to local files (as before)
    scraped_files = []
    for url in WEBSITE_URLS:
        text = scrape_website(url)
        if text:
            filename = url.replace("https://", "").replace("/", "_").replace(".", "_") + ".txt"
            scraped_files.append(save_text_to_file(text, filename))

    # 2. Load documents from scraped files
    documents = load_documents_from_files()
    print(f"Loaded {len(documents)} documents from text files.")

    # 3. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    # 4. Initialize Google Generative AI Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
    # 5. Initialize Pinecone and create/connect to index
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

    if INDEX_NAME not in [index.name for index in pc.list_indexes()]:
        print(f"Creating Pinecone index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSIONS,
            metric=METRIC,
            spec=ServerlessSpec(cloud='aws', region='us-east-1') # Confirmed from your screenshot
        )
        print(f"Index '{INDEX_NAME}' created.")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")

    # 6. Upload chunks to Pinecone
    print(f"Uploading {len(chunks)} chunks to Pinecone index '{INDEX_NAME}'...")
    # PineconeVectorStore.from_documents will handle the embedding and upload
    vectorstore = PineconeVectorStore.from_documents(
        chunks, embeddings, index_name=INDEX_NAME
    )
    print("Data ingestion to Pinecone complete!")

if __name__ == "__main__":
    ingest_data_to_pinecone()