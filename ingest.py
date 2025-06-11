import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# LangChain components
from langchain_community.document_loaders import TextLoader # Or HtmlLoader if you load local files
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables (your Google API key)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file. Please set it.")

# --- Configuration ---
# List of URLs to scrape from your website
WEBSITE_URLS = [
    "https://theapexarena.framer.website/", # Replace with your actual Framer URL
    "https://theapexarena.framer.website/taylor-swift",
    "https://theapexarena.framer.website/brunch", # Replace with your actual Framer URL
    "https://theapexarena.framer.website/trade-show",
    "https://theapexarena.framer.website/basketball", # Replace with your actual Framer URL
    "https://theapexarena.framer.website/community-gathering",
    "https://theapexarena.framer.website/seminar", # Replace with your actual Framer URL
    "https://theapexarena.framer.website/renting",
    "https://theapexarena.framer.website/about-us", # Replace with your actual Framer URL
    "https://theapexarena.framer.website/projects/le-blink",
    "https://theapexarena.framer.website/projects/schlong", # Replace with your actual Framer URL
    "https://theapexarena.framer.website/projects/vintage-everything",
    "https://theapexarena.framer.website/projects/senseya", # Replace with your actual Framer URL
    "https://theapexarena.framer.website/projects", # Example for an events page
    # Add more URLs for your services, FAQ, contact, etc.
]
# Directory to store scraped text (optional, but good for debugging)
TEXT_DATA_DIR = "text_data"
# Directory where ChromaDB will store its persistent data
CHROMA_DB_DIR = "chroma_db"

# --- Web Scraper Function ---
def scrape_website_content(urls):
    print("Starting website scraping...")
    documents = []
    os.makedirs(TEXT_DATA_DIR, exist_ok=True)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    } # ADD THESE LINES

    for url in urls:
        try:
            print(f"Scraping: {url}")
            response = requests.get(url, timeout=10)
            response.raise_for_status() # Raise an exception for HTTP errors

            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract main content (adjust selectors based on your Framer site's HTML)
            # Common selectors: 'main', 'article', 'div.content-section'
            # Inspect your Framer site's HTML to find the best selector
            main_content = soup.find('body') # Often 'body' or a specific content div
            if main_content:
                text_content = main_content.get_text(separator='\n', strip=True)
                # Filter out common script/style tags if they get through
                text_content = "\n".join(
                    [line for line in text_content.splitlines() if line.strip() and not line.strip().startswith(('var', 'function', '{', '}', '//'))]
                )

                # Save for inspection
                filename = os.path.join(TEXT_DATA_DIR, f"{url.replace('https://', '').replace('/', '_')}.txt")
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(text_content)
                documents.append({"page_content": text_content, "metadata": {"source": url}})
                print(f"  - Scraped {len(text_content)} characters from {url}")
            else:
                print(f"  - No main content found for {url}")

        except requests.exceptions.RequestException as e:
            print(f"  - Error scraping {url}: {e}")
        except Exception as e:
            print(f"  - An unexpected error occurred with {url}: {e}")
    print("Scraping finished.")
    return documents

# --- Main Ingestion Logic ---
def ingest_data():
    print("Starting data ingestion process...")

    # 1. Scrape content
    raw_documents = scrape_website_content(WEBSITE_URLS)
    if not raw_documents:
        print("No documents scraped. Exiting ingestion.")
        return

    # 2. Split documents into chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Characters per chunk
        chunk_overlap=200,    # Overlap between chunks to maintain context
        length_function=len,
        is_separator_regex=False,
    )
    texts = []
    for doc in raw_documents:
        # LangChain's text splitter expects Document objects
        # Convert raw_documents to LangChain Document objects if not already
        from langchain.docstore.document import Document
        doc_obj = Document(page_content=doc['page_content'], metadata=doc['metadata'])
        texts.extend(text_splitter.split_documents([doc_obj]))
    print(f"Split into {len(texts)} chunks.")

    # 3. Create embeddings
    print("Creating embeddings (this may take a while)...")
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

    # 4. Store in ChromaDB
    print(f"Storing embeddings in ChromaDB at {CHROMA_DB_DIR}...")
    # This will create/load the collection and add the documents
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings_model,
        persist_directory=CHROMA_DB_DIR
    )
    vectordb.persist() # Ensures data is written to disk
    print("Data ingestion complete. ChromaDB updated.")

if __name__ == "__main__":
    ingest_data()