from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from qdrant_client.http.models import VectorParams, Distance

try:
    # Initialize Qdrant client
    qdrant_client = QdrantClient(host='localhost', port=6333)
    
    # Test connection by getting collections
    collections = qdrant_client.get_collections()
    print("Successfully connected to Qdrant!")
    print("Existing collections:")
    for collection in collections.collections:
        print(f"- {collection.name}")
except Exception as e:
    print(f"Error connecting to Qdrant: {e}")
