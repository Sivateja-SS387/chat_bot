# Product Information Vector Ingestion Pipeline

## Overview
This Python script provides a robust, incremental data ingestion pipeline for transforming product information into vector embeddings and storing them in a Qdrant vector database.

## Features
- 🔄 Incremental Data Ingestion
- 🧠 Advanced Embedding Generation
- 🛡️ Error Handling and Validation
- 💾 PostgreSQL and Qdrant Integration

## Prerequisites
- Python 3.8+
- PostgreSQL
Docker
- Qdrant
- Required Python Packages:
  - `sentence-transformers`
  - `qdrant-client`
  - `psycopg2`
  - `numpy`
  - `uuid`

### Environment Setup
1. Clone the repository
2. Run command `docker-compose up`
3. You can use `docker-compose down` to stop the containers


## Configuration
### Database Connection
Configure your PostgreSQL and Qdrant connection parameters in the script:
```python
connection_parameters = {
    'host': 'localhost',
    'database': 'your_database',
    'user': 'your_username',
    'password': 'your_password'
}
```

### Embedding Model
Currently using: `sentence-transformers/all-MiniLM-L6-v2`
- Vector Dimension: 384
- Distance Metric: Cosine Similarity

## Key Components

### 1. Embedding Generation
- Uses Sentence Transformers for generating high-quality embeddings
- Validates embedding dimensions
- Supports incremental embedding generation

### 2. Qdrant Collection Management
- Dynamically creates or verifies Qdrant collections
- Handles vector configuration
- Supports incremental point insertion

### 3. Ingestion Logging
- Tracks ingestion progress in a dedicated PostgreSQL table
- Logs:
  - Last processed record ID
  - Number of records processed
  - Timestamp of ingestion

## Ingestion Process Workflow
1. Initialize embedding model
2. Check/Create Qdrant collection
3. Retrieve last processed record
4. Fetch new records from PostgreSQL
5. Generate embeddings
6. Upsert points to Qdrant
7. Log ingestion progress

## Error Handling
- Comprehensive logging of errors
- Graceful error recovery
- Detailed diagnostic information

## Performance Considerations
- Batch processing (default: 1000 records)
- Duplicate point elimination
- Configurable batch size

## Logging
Uses Python's `logging` module with multiple log levels:
- INFO: General process information
- DEBUG: Detailed diagnostic information
- ERROR: Critical errors and exceptions

## Installation and Setup

### Prerequisites
- Python 3.8+
- PostgreSQL 12+
- Qdrant Vector Database

### Required Packages
Install the following Python packages using pip:

No Need of manual installation -Docker will run the requirements.txt file and install the packages for you


### Running the Ingestion Script
```bash
python ingest.py
```

### Logging
- Logs are generated in `qdrant_ingest_debug.log`
- Logging level is set to DEBUG
- Logs include timestamps, log levels, and detailed messages

### Features
- Incremental data ingestion
- Embedding generation using BAAI/bge-large-en model
- Error handling and validation
- PostgreSQL and Qdrant integration

### Troubleshooting
- Verify PostgreSQL and Qdrant connection details
- Check `qdrant_ingest_debug.log` for detailed error messages
- Ensure all required packages are installed

## Usage
```bash
python ingest.py
```

## Customization
- Modify `connection_parameters` for your database
- Adjust embedding model in `initialize_embedding_model()`
- Configure batch size in `fetch_product_info()`

## Troubleshooting
- Check log files for detailed error information
- Ensure database and Qdrant are running
- Verify network connectivity

## Security
- Avoid hardcoding sensitive credentials
- Use environment variables or secure credential management

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request


---

**Note**: This script is part of a vector search implementation for product information retrieval.
