import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from typing import List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_constitution_from_web(url: str) -> List[Document]:
    """Load the constitution from the official website with robust error handling"""
    try:
        # headers 
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        logger.info(f"Fetching constitution from {url}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        
        logger.info("Parsing HTML content")
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # main content 
        content_div = soup.find('div', {'class': 'text_block'}) or soup.find('article') or soup.find('main')
        
        if not content_div:
            logger.warning("Could not find main content div, falling back to body")
            content_div = soup.body
        
        # Extract and clean text
        logger.info("Extracting text content")
        text = content_div.get_text(separator='\n', strip=True)
        
        # Basic cleaning
        text = '\n'.join(line for line in text.split('\n') if line.strip())
        
       
        logger.info("Splitting into articles/sections")
        articles = [art for art in text.split('\n\n') if art.strip()]
        
        documents = []
        for i, article in enumerate(articles, start=1):
            if len(article) < 20:
                continue
                
            documents.append(Document(
                page_content=article,
                metadata={
                    "source": "constitution",
                    "article_number": i,
                    "url": url
                }
            ))
            
        logger.info(f"Successfully loaded {len(documents)} articles from constitution")
        return documents
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error loading constitution: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error processing constitution: {e}")
        return []

def load_constitution_fallback() -> List[Document]:
    """Fallback constitution data if web scraping fails"""
    logger.warning("Using fallback constitution data")
    from langchain.schema import Document
    return [
        Document(
            page_content="The Constitution of the Republic of Kazakhstan is the supreme law of Kazakhstan.",
            metadata={"source": "fallback", "note": "Basic constitutional information"}
        )
    ]

def get_constitution_documents(url: str) -> List[Document]:
    """Main function to get constitution documents with fallback"""
    docs = load_constitution_from_web(url)
    if not docs:
        docs = load_constitution_fallback()
    return docs