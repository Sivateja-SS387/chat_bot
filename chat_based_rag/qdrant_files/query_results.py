from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import numpy as np
import sys

def create_qdrant_client(host='localhost', port=6333, timeout=10):
    """
    Create and validate Qdrant client connection
    
    Args:
        host (str): Qdrant server host
        port (int): Qdrant server port
        timeout (int): Connection timeout in seconds
    
    Returns:
        QdrantClient: Configured Qdrant client
    """
    try:
        client = QdrantClient(
            host=host, 
            port=port, 
            timeout=timeout
        )
        
        # Validate connection
        client.get_collections()
        return client
    
    except Exception as e:
        print(f"Failed to connect to Qdrant: {e}")
        raise

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Qdrant client with robust connection handling
try:
    qdrant_client = create_qdrant_client()
except Exception:
    sys.exit(1)

def semantic_search(query, top_k=5, similarity_threshold=0.2):
    """
    Perform semantic search with optional similarity threshold
    
    Args:
        query (str): Search query
        top_k (int): Number of top results to return
        similarity_threshold (float): Minimum similarity score to consider
    
    Returns:
        List of filtered and sorted search results
    """
    try:
        # Convert query to embedding
        query_embedding = model.encode(query)
        
        # Perform search in Qdrant
        search_results = qdrant_client.search(
            collection_name="product_info_collection",
            query_vector=query_embedding,
            limit=top_k * 2  # Fetch more to allow filtering
        )
        
        # Process and filter results
        results = []
        for hit in search_results:
            # Apply similarity threshold
            if hit.score >= similarity_threshold:
                result = hit.payload.copy()
                result['similarity_score'] = hit.score
                results.append(result)
        
        # Sort results by similarity score
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return results[:top_k]
    
    except Exception as e:
        print(f"Search failed: {e}")
        return []

def format_result(result):
    """
    Format a single search result for readable output
    
    Args:
        result (dict): Single search result
    
    Returns:
        str: Formatted result string
    """
    formatted = "--- Search Result ---\n"
    for key, value in result.items():
        if key != 'id':  # Exclude raw ID
            formatted += f"{key.replace('_', ' ').title()}: {value}\n"
    return formatted

def main():
    # Specific query for Aspirin Details
    query = "Aspirin Details"
    
    # Perform search
    results = semantic_search(query)
    
    # Print results
    if results:
        print(f"Search Results for '{query}':")
        for result in results:
            print(format_result(result))
            print(f"Similarity Score: {result['similarity_score']:.4f}\n")
    else:
        print("No results found.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Ensure clean client shutdown if possible
        try:
            if 'qdrant_client' in locals():
                qdrant_client.close()
        except Exception:
            pass
