import psycopg2
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams
from qdrant_client.http.models import Distance, Filter, FieldCondition, Match
import uuid
import sys
import traceback
import logging
import numpy as np
import datetime
from langchain_huggingface import HuggingFaceEmbeddings
import os

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qdrant_ingest_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Define your connection parameters
connection_parameters = {
    'host': os.getenv('POSTGRES_HOST', 'drug-data-analysis_a04f6f-postgres-1'),
    'port': os.getenv('POSTGRES_PORT', '5432'),
    'database': os.getenv('POSTGRES_DB', 'mms_dbs'),
    'user': os.getenv('POSTGRES_USER', 'postgres'),
    'password': os.getenv('POSTGRES_PASSWORD', 'postgres')
}

# Initialize the model for generating embeddings
def initialize_embedding_model():
    """
    Initialize and validate the embedding model

    Returns:
        SentenceTransformer: Initialized model
    """
    try:
        model = SentenceTransformer('BAAI/bge-large-en')
        
        test_text = "Validate embedding generation"
        test_embedding = model.encode(test_text)

        logger.info("Embedding model initialized successfully")
        logger.debug(f"Test embedding dimension: {len(test_embedding)}")

        return model
    except Exception as e:
        logger.error(f"Failed to initialize Sentence Transformer: {e}")
        raise

# Global model initialization
model = initialize_embedding_model()

# Initialize Qdrant client with detailed logging
def initialize_qdrant_client():
    """
    Initialize Qdrant client with comprehensive diagnostics

    Returns:
        QdrantClient: Initialized Qdrant client
    """
    try:
        qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
        qdrant_port = int(os.getenv('QDRANT_PORT', '6334'))
        client = QdrantClient(host=qdrant_host, port=qdrant_port)
        try:
            cluster_info = client.cluster_info()
            logger.info("Qdrant cluster information:")
            logger.info(f"Cluster status: {cluster_info}")
        except Exception as cluster_error:
            logger.warning(f"Could not retrieve cluster info: {cluster_error}")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant client: {e}")
        raise

# Global Qdrant client initialization
qdrant_client = initialize_qdrant_client()

# Define collection name
collection_name = "drug_labels_collection_data_final"

 
def get_last_processed_id(collection_name):
    """
    Retrieve the last processed ID from the log table
   
    Args:
        collection_name (str): Name of the collection to check
   
    Returns:
        int: Last processed ID or 0 if no previous ingestion
    """
    connection = None
    try:
        connection = psycopg2.connect(**connection_parameters)
        cursor = connection.cursor()
       
        # Get the last processed ID for this specific collection
        query = """
        SELECT last_processed_id
        FROM qdrant_ingestion_log
        WHERE collection_name = %s
        ORDER BY processed_at DESC
        LIMIT 1
        """
       
        cursor.execute(query, (collection_name,))
        result = cursor.fetchone()
       
        # Return last processed ID or 0
        return result[0] if result else 0
   
    except Exception as e:
        logger.error(f"Error retrieving last processed ID: {e}")
        return 0
   
    finally:
        if connection:
            connection.close()
 

# Function to generate embeddings with comprehensive validation
def generate_embeddings(data):
    """
    Generate embeddings with detailed validation

    Args:
        data (list): Text data to generate embeddings for

    Returns:
        numpy.ndarray: Generated embeddings
    """
    try:
        if not data:
            logger.warning("Empty data provided for embedding generation")
            return np.array([])

        embeddings = model.encode(data)

        logger.info(f"Generated {len(embeddings)} embeddings")
        logger.debug(f"Embedding shape: {embeddings.shape}")

        if embeddings.shape[1] != 384:
            logger.error(f"Unexpected embedding dimension: {embeddings.shape[1]}")
            raise ValueError(f"Embedding dimension must be 384, got {embeddings.shape[1]}")

        return embeddings

    except Exception as e:
        logger.error(f"Embedding generation error: {e}")
        traceback.print_exc()
        raise

# Function to create or update a Qdrant collection with specific vector configuration
def create_collection(collection_name):
    """
    Create or update a Qdrant collection with specific vector configuration

    Args:
        collection_name (str): Name of the collection to create/update
    """
    try:
        # Check existing collections
        collections = qdrant_client.get_collections()
        existing_collections = [col.name for col in collections.collections]

        logger.info(f"Existing collections: {existing_collections}")

        # If collection doesn't exist, create it
        if collection_name not in existing_collections:
            logger.info(f"Creating new collection: {collection_name}")
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE
                )
            )
        else:
            logger.info(f"Collection {collection_name} already exists. Skipping creation.")

        # Verify collection configuration
        try:
            collection_info = qdrant_client.get_collection(collection_name)
            logger.debug(f"Collection details: Vector size={collection_info.config.params.vectors.size}")
        except Exception as info_error:
            logger.error(f"Error retrieving collection info: {info_error}")

    except Exception as e:
        logger.error(f"Collection configuration error: {e}")
        traceback.print_exc()
        raise

# Function to add data to Qdrant collection
def add_data_to_collection(data, embeddings, collection_name):
    """
    Add data points to Qdrant collection with comprehensive error handling

    Args:
        data (list): Raw data to be inserted
        embeddings (numpy.ndarray): Generated embeddings
        collection_name (str): Name of the Qdrant collection
    """
    if len(data) == 0 or len(embeddings) == 0:
        logger.warning("No data or embeddings to insert")
        return

    unique_points = []
    for item, embedding in zip(data, embeddings):
        text_id = str(uuid.uuid4())
        payload = {
            'id':item[0],
            'spl_set_id': item[1],
            'brand_name': item[2],
            'generic_name': item[3],
            'product_type': item[4],
            'purpose': item[5],
            'indications_and_usage': item[6],
            'description': item[7]
        }
        # print(list(item),end='\n')

        # print(list(embedding),end='\n')
        point = PointStruct(
            id=text_id,
            vector=embedding,
            payload=payload
        )
        unique_points.append(point)

    try:
        logger.info(f"Preparing to upsert {len(unique_points)} points to collection: {collection_name}")

        # Perform upsert operation
        operation_info = qdrant_client.upsert(
            collection_name=collection_name,
            wait=True,
            points=unique_points
        )

        # Get current collection info to verify points
        collection_info = qdrant_client.get_collection(collection_name)
        logger.info(f"Total points in collection after upsert: {collection_info.points_count}")

        logger.info(f"Successfully upserted {len(unique_points)} points to collection: {collection_name}")

    except Exception as e:
        logger.error(f"Qdrant upsert error: {e}")
        logger.error(f"Error details: {type(e).__name__}")

        try:
            collection_info = qdrant_client.get_collection(collection_name)
            logger.error(f"Collection points count: {collection_info.points_count}")
        except Exception as info_error:
            logger.error(f"Could not retrieve collection info: {info_error}")

        raise

def log_ingestion_progress(last_processed_id, processed_count, collection_name):
    """
    Log the progress of data ingestion
   
    Args:
        last_processed_id (int): ID of the last processed record
        processed_count (int): Number of records processed
        collection_name (str): Name of the Qdrant collection
    """
    connection = None
    try:
        connection = psycopg2.connect(**connection_parameters)
        cursor = connection.cursor()
       
        # Insert ingestion log
        insert_query = """
        INSERT INTO qdrant_ingestion_log
        (last_processed_id, processed_count, collection_name)
        VALUES (%s, %s, %s)
        """
       
        cursor.execute(insert_query, (last_processed_id, processed_count, collection_name))
        connection.commit()
       
        logger.info(f"Logged ingestion progress: Last ID {last_processed_id}, Processed {processed_count} records")
   
    except Exception as e:
        logger.error(f"Error logging ingestion progress: {e}")
        connection.rollback()
   
    finally:
        if connection:
            connection.close()
 
def get_total_ingested_records(collection_name):
    """
    Get total number of records ingested for a specific collection
   
    Args:
        collection_name (str): Name of the collection
   
    Returns:
        int: Total number of records ingested
    """
    connection = None
    try:
        connection = psycopg2.connect(**connection_parameters)
        cursor = connection.cursor()
       
        # Sum of processed records for this collection
        query = """
        SELECT COALESCE(SUM(processed_count), 0)
        FROM qdrant_ingestion_log
        WHERE collection_name = %s
        """
       
        cursor.execute(query, (collection_name,))
        total_records = cursor.fetchone()[0]
       
        logger.info(f"Total records already ingested for {collection_name}: {total_records}")
        return total_records
   
    except Exception as e:
        logger.error(f"Error retrieving total ingested records: {e}")
        return 0
   
    finally:
        if connection:
            connection.close()
 

def search_with_filters(query, collection_name, filter_conditions=None, limit=10):
    """
    Perform a vector search with filters in Qdrant.

    Args:
        query (str): The query string to search for.
        collection_name (str): The Qdrant collection name.
        filter_conditions (dict, optional): A dictionary of metadata filters (default is None).
        limit (int, optional): The number of search results to return (default is 10).

    Returns:
        List: List of search results with metadata.
    """
    try:
        # Generate embedding for the query
        query_embedding = model.encode([query])[0]  # Encode the query

        # Define the filter for metadata (if any)
        qdrant_filter = None
        if filter_conditions:
            filter_items = [FieldCondition(key=key, match=Match(value=value)) for key, value in filter_conditions.items()]
            qdrant_filter = Filter(must=filter_items)

        # Perform search with filter and vector similarity
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
            limit=limit,
            filter=qdrant_filter  # Apply the filter conditions
        )

        logger.info(f"Search completed with {len(search_results)} results for query: {query}")
        return search_results

    except Exception as e:
        logger.error(f"Error during search with filters: {e}")
        traceback.print_exc()
        return []


# Function to create a log table to track data ingestion progress
def create_ingestion_log_table():
    """
    Create a log table to track data ingestion progress
    """
    connection = None
    try:
        connection = psycopg2.connect(**connection_parameters)
        cursor = connection.cursor()

        # Create table if not exists
        create_table_query = """
        CREATE TABLE IF NOT EXISTS qdrant_ingestion_log (
            id SERIAL PRIMARY KEY,
            last_processed_id INTEGER NOT NULL,
            processed_count INTEGER NOT NULL,
            processed_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
            collection_name VARCHAR(255) NOT NULL
        )
        """

        cursor.execute(create_table_query)
        connection.commit()
        logger.info("Qdrant ingestion log table created or verified")

    except Exception as e:
        logger.error(f"Error creating ingestion log table: {e}")
        raise

    finally:
        if connection:
            connection.close()

from langchain_community.embeddings import HuggingFaceEmbeddings

hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def concatenate_fields(row):

    print(f"ID: {row[0]}, SPL Set ID: {row[1]}, Brand Name: {row[2]}, Generic Name: {row[3]}, Product Type: {row[4]}, Purpose: {row[5]}, Indications: {row[6]}, Description: {row[7]}")

    return f"Brand Name: {row[2]}, Generic Name: {row[3]}, Product Type: {row[4]}, Purpose: {row[5]}, Indications: {row[6]}, Description: {row[7]}"


# Function to fetch product information and ingest it into Qdrant
def fetch_drug_labels():
    """
    Fetch product information from PostgreSQL and ingest into Qdrant with incremental processing

    Returns:
        str or None: Name of the collection if successful, None otherwise
    """
    connection = None
    try:
        # Ensure log table exists
        create_ingestion_log_table()

        try:
            collection_info = qdrant_client.get_collection(collection_name)
            logger.info(f"Existing collection points: {collection_info.points_count}")
        except Exception:
            logger.info(f"New collection: {collection_name}")

        # Get last processed ID for this specific collection

        last_processed_id = get_last_processed_id(collection_name)
        # last_processed_id = 0  # For simplicity, starting from 0 (you can customize this)

        if last_processed_id == 0:
            logger.info(f"No previous ingestion for {collection_name}. Starting from the first record.")
            last_processed_id = -1  # Start from the first record
       
        connection = psycopg2.connect(**connection_parameters)

        # Create collection if not exists
        create_collection(collection_name)

        cursor = connection.cursor()

        # Modified query to process batches
        query = """
            SELECT id, spl_set_id, brand_name, generic_name, product_type,
            purpose, indications_and_usage, description FROM drug_labels
            WHERE id > %s AND brand_name != '' AND generic_name != ''
            ORDER BY id
            LIMIT 1000
        """

        cursor.execute(query, (last_processed_id,))
        data = cursor.fetchall()


        if not data:
            logger.warning("No new data found in drug_labels table")
            return None

        
        # data = []
        # for row in rows:
        #     data.append(row)  # Populate data with relevant columns
        # text_data = [row[6] for row in rows]
        
        concatenated_data = [concatenate_fields(row) for row in data]
        embeddings = hf_embeddings.embed_documents(concatenated_data)
    

        # embeddings = generate_embeddings([item[2] for item in data])  
        # embeddings = generate_embeddings(text_data)  # Assuming product descriptions are in column 3
        # print([item[2] for item in data])
        # print(embeddings)

        add_data_to_collection(data, embeddings, collection_name)

        # Update the last processed ID in the log table (for incremental processing)
        last_id = data[-1][0]  # Assuming the first column is the ID
        log_ingestion_progress(last_id, len(data), collection_name)
       
        # Log total ingestion status
        total_ingested_after = get_total_ingested_records(collection_name)
        logger.info(f"Total records ingested for {collection_name}: {total_ingested_after}")

        logger.info(f"Successfully ingested data for collection {collection_name}")
        return collection_name

    except Exception as e:
        logger.error(f"Error during data ingestion: {e}")
        traceback.print_exc()
        return None

    finally:
        if connection:
            connection.close()

# Fetch and ingest product information into Qdrant
fetch_drug_labels()
