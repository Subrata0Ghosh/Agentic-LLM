import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import uuid
import nltk
import logging
import os

# Ensure NLTK punkt is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('punkt')

class VectorMemory:
    def __init__(self, db_path="data/chroma_db"):
        self.logger = logging.getLogger("VectorMemory")
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Force CPU on Hugging Face Spaces to avoid ZeroGPU thread allocation errors
        device = "cpu" if "SPACE_ID" in os.environ else None
        if device:
            print("[VectorMemory] Running on Hugging Face Spaces - forcing CPU execution")
            
        # Explicitly use a powerful but fast sentence-transformer
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
            device=device
        )
        
        # Create or get a collection for long-term memory
        self.collection = self.client.get_or_create_collection(
            name="agentic_memory",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def _chunk_text(self, text, max_sentences=5):
        """Splits text into chunks based on sentences to preserve semantic meaning."""
        sentences = nltk.tokenize.sent_tokenize(text)
        chunks = []
        for i in range(0, len(sentences), max_sentences):
            chunks.append(" ".join(sentences[i:i + max_sentences]))
        return chunks

    def store(self, text, source="unknown", additional_metadata=None):
        """Stores text chunks into the vector database with rich metadata."""
        if not text.strip():
            return
            
        chunks = self._chunk_text(text)
        
        ids = [str(uuid.uuid4()) for _ in chunks]
        
        # Build base metadata
        base_meta = {"source": source}
        if additional_metadata:
            # Ensure metadata values are strings, ints, or floats
            for k, v in additional_metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    base_meta[k] = v
                else:
                    base_meta[k] = str(v)
                    
        metadatas = [base_meta.copy() for _ in chunks]
        
        self.collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        self.logger.info(f"Stored {len(chunks)} memory chunks from '{source}'.")
        print(f"Stored {len(chunks)} memory chunks from '{source}'.")

    def retrieve(self, query, top_k=3, filter_metadata=None):
        """Retrieves the most relevant memory chunks for a given query."""
        query_params = {
            "query_texts": [query],
            "n_results": top_k,
            "include": ["documents", "distances"]
        }
        if filter_metadata:
            query_params["where"] = filter_metadata
            
        results = self.collection.query(**query_params)
        
        if not results['documents'][0]:
            return [], []
            
        return results['documents'][0], results['distances'][0]
        
    def get_all_topics(self):
        """Helper to get a summary of what's in memory for the UI Inspector."""
        # A simple approach: grab everything and extract unique sources
        try:
            results = self.collection.get(include=["metadatas"])
            sources = set()
            for meta in results['metadatas']:
                if meta and 'source' in meta:
                    sources.add(meta['source'])
            return list(sources)
        except Exception as e:
            print(f"Error getting topics: {e}")
            return []

    def get_context_string(self, query, top_k=3, threshold=0.4):
        """Retrieves memories and formats them. Returns (context_string, has_good_match)."""
        docs, dists = self.retrieve(query, top_k=top_k)
        if not docs:
            return "", False
            
        has_good_match = False
        if dists and dists[0] < threshold:
            has_good_match = True
            
        context = "### RELEVANT MEMORIES (RAG CONTEXT) ###\n"
        # Only include facts that are somewhat close to the query
        for i, fact in enumerate(docs):
            if dists[i] < threshold + 0.3:
                context += f"Fact {i+1}: {fact}\n"
        context += "### END MEMORIES ###\n\n"
        return context, has_good_match
