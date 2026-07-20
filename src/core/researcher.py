import arxiv
from vector_memory import VectorMemory

def research_topic(topic, max_results=2):
    """
    Searches ArXiv for the given topic, downloads the metadata and abstract,
    and stores it in the vector memory with structured metadata.
    """
    print(f"Researching ArXiv for: '{topic}'")
    
    # Construct the default API client.
    client = arxiv.Client()
    
    # Search for the topic
    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    memory = VectorMemory()
    papers_read = 0
    
    for result in client.results(search):
        title = result.title
        abstract = result.summary
        authors = ", ".join([author.name for author in result.authors])
        published = result.published.strftime("%Y-%m-%d")
        
        # Format the text to store with clear section headers
        text_to_store = f"Paper Title: {title}\n\nAuthors: {authors}\n\nPublished: {published}\n\nAbstract:\n{abstract}"
        
        # Build additional metadata for the chunks
        meta = {
            "title": title,
            "topic": topic,
            "published_date": published,
            "is_paper": True
        }
        
        # Store in vector database
        memory.store(
            text=text_to_store, 
            source=f"arxiv_{result.get_short_id()}",
            additional_metadata=meta
        )
        papers_read += 1
        print(f"Read and memorized: {title}")
        
    return f"Successfully read and memorized {papers_read} research papers on '{topic}'."

if __name__ == "__main__":
    # Test the researcher
    research_topic("Large Language Models")
