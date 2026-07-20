import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
import re

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def clean_text(text):
    # Remove excessive newlines and whitespace
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def search_and_scrape(query, num_results=5):
    print(f"Searching DuckDuckGo for: '{query}'")
    
    scraped_data = []
    
    try:
        # Perform DuckDuckGo search
        results = []
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        for res in results:
            url = res.get('href')
            snippet = res.get('body') or res.get('snippet') or ""
            if not url:
                continue
            print(f"Scraping {url}...")
            scraped_text = ""
            try:
                # Disable SSL verify for some university websites that have broken certificates
                response = requests.get(url, headers=headers, timeout=8, verify=False)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extract text from paragraphs
                    paragraphs = soup.find_all('p')
                    text_content = " ".join([p.get_text() for p in paragraphs])
                    
                    cleaned = clean_text(text_content)
                    if len(cleaned) > 120:  # Only keep substantial text
                        scraped_text = cleaned
                else:
                    print(f"Failed to fetch {url}, status code: {response.status_code}")
            except Exception as e:
                print(f"Error scraping {url}: {e}")
                
            if scraped_text:
                scraped_data.append(scraped_text)
            elif snippet:
                print(f"Scrape failed/blocked for {url}. Falling back to search snippet.")
                scraped_data.append(f"[Search Snippet: {url}] {snippet}")
                
    except Exception as e:
        print(f"Error during search: {e}")
        
    final_text = "\n\n".join(scraped_data)
    
    if final_text:
        # Append to our corpus
        with open("data/ai_corpus_cleaned.txt", "a", encoding="utf-8") as f:
            f.write("\n\n" + final_text)
        print(f"Added {len(final_text)} characters to corpus.")
        
    return final_text
