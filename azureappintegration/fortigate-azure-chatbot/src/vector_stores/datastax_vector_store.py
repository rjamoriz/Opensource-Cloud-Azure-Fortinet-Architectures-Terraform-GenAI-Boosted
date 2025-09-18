"""
DataStax Vector Database Integration
Connect to DataStax Astra DB for vector storage and retrieval
"""

import os
import logging
import json
import requests
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import streamlit as st

logger = logging.getLogger(__name__)

class DataStaxVectorStore:
    """DataStax Astra DB Vector Store implementation"""
    
    def __init__(self, api_endpoint: str = None, api_key: str = None, keyspace: str = "default_keyspace"):
        self.api_endpoint = api_endpoint or os.getenv('DATASTAX_API_ENDPOINT')
        self.api_key = api_key or os.getenv('DATASTAX_API_KEY')
        self.keyspace = keyspace
        self.collection_name = "fortigate_documents"
        self.session = requests.Session()
        self._setup_session()
    
    def _setup_session(self):
        """Setup HTTP session with authentication"""
        if self.api_key:
            self.session.headers.update({
                'X-Cassandra-Token': self.api_key,
                'Content-Type': 'application/json'
            })
    
    def test_connection(self) -> Tuple[bool, str]:
        """Test connection to DataStax Astra DB"""
        try:
            if not self.api_endpoint or not self.api_key:
                return False, "Missing API endpoint or API key"
            
            # Test connection with a simple keyspace query
            url = f"{self.api_endpoint}/api/rest/v2/keyspaces"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                keyspaces = response.json().get('data', [])
                return True, f"Connected successfully. Available keyspaces: {len(keyspaces)}"
            else:
                return False, f"Connection failed: {response.status_code} - {response.text}"
                
        except Exception as e:
            return False, f"Connection error: {str(e)}"
    
    def create_collection(self) -> Tuple[bool, str]:
        """Create vector collection if it doesn't exist"""
        try:
            # Check if collection exists
            url = f"{self.api_endpoint}/api/json/v1/{self.keyspace}/{self.collection_name}"
            response = self.session.get(url)
            
            if response.status_code == 200:
                return True, "Collection already exists"
            
            # Create collection with vector support
            create_url = f"{self.api_endpoint}/api/json/v1/{self.keyspace}"
            collection_schema = {
                "createCollection": {
                    "name": self.collection_name,
                    "options": {
                        "vector": {
                            "dimension": 1536,  # OpenAI embedding dimension
                            "metric": "cosine"
                        }
                    }
                }
            }
            
            response = self.session.post(create_url, json=collection_schema)
            
            if response.status_code in [200, 201]:
                return True, "Collection created successfully"
            else:
                return False, f"Failed to create collection: {response.status_code} - {response.text}"
                
        except Exception as e:
            return False, f"Error creating collection: {str(e)}"
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Add documents with embeddings to the vector store"""
        try:
            if not documents:
                return False, "No documents provided"
            
            # Ensure collection exists
            success, message = self.create_collection()
            if not success:
                return False, f"Collection setup failed: {message}"
            
            url = f"{self.api_endpoint}/api/json/v1/{self.keyspace}/{self.collection_name}"
            
            # Process documents in batches
            batch_size = 20
            total_added = 0
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                # Prepare batch insert
                insert_data = {
                    "insertMany": {
                        "documents": []
                    }
                }
                
                for doc in batch:
                    document = {
                        "_id": doc.get('id', f"doc_{datetime.now().timestamp()}"),
                        "content": doc.get('content', ''),
                        "metadata": doc.get('metadata', {}),
                        "timestamp": datetime.now().isoformat(),
                        "$vector": doc.get('embedding', [])
                    }
                    insert_data["insertMany"]["documents"].append(document)
                
                response = self.session.post(url, json=insert_data)
                
                if response.status_code in [200, 201]:
                    result = response.json()
                    total_added += len(result.get('status', {}).get('insertedIds', []))
                else:
                    logger.error(f"Batch insert failed: {response.status_code} - {response.text}")
            
            return True, f"Successfully added {total_added} documents"
            
        except Exception as e:
            return False, f"Error adding documents: {str(e)}"
    
    def similarity_search(self, query_embedding: List[float], limit: int = 5) -> Tuple[bool, List[Dict[str, Any]]]:
        """Perform similarity search using vector embeddings"""
        try:
            url = f"{self.api_endpoint}/api/json/v1/{self.keyspace}/{self.collection_name}"
            
            search_query = {
                "find": {
                    "sort": {
                        "$vector": query_embedding
                    },
                    "limit": limit,
                    "includeSimilarity": True
                }
            }
            
            response = self.session.post(url, json=search_query)
            
            if response.status_code == 200:
                result = response.json()
                documents = result.get('data', {}).get('documents', [])
                
                # Format results
                formatted_results = []
                for doc in documents:
                    formatted_results.append({
                        'id': doc.get('_id'),
                        'content': doc.get('content', ''),
                        'metadata': doc.get('metadata', {}),
                        'similarity': doc.get('$similarity', 0.0),
                        'timestamp': doc.get('timestamp')
                    })
                
                return True, formatted_results
            else:
                return False, []
                
        except Exception as e:
            logger.error(f"Similarity search error: {e}")
            return False, []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            url = f"{self.api_endpoint}/api/json/v1/{self.keyspace}/{self.collection_name}"
            
            # Count documents
            count_query = {
                "countDocuments": {}
            }
            
            response = self.session.post(url, json=count_query)
            
            if response.status_code == 200:
                result = response.json()
                document_count = result.get('status', {}).get('count', 0)
                
                return {
                    'total_documents': document_count,
                    'collection_name': self.collection_name,
                    'keyspace': self.keyspace,
                    'vector_dimension': 1536,
                    'status': 'active'
                }
            else:
                return {
                    'total_documents': 0,
                    'collection_name': self.collection_name,
                    'keyspace': self.keyspace,
                    'status': 'error'
                }
                
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {
                'total_documents': 0,
                'status': 'error',
                'error': str(e)
            }
    
    def delete_documents(self, document_ids: List[str]) -> Tuple[bool, str]:
        """Delete documents by IDs"""
        try:
            url = f"{self.api_endpoint}/api/json/v1/{self.keyspace}/{self.collection_name}"
            
            delete_query = {
                "deleteMany": {
                    "filter": {
                        "_id": {"$in": document_ids}
                    }
                }
            }
            
            response = self.session.post(url, json=delete_query)
            
            if response.status_code == 200:
                result = response.json()
                deleted_count = result.get('status', {}).get('deletedCount', 0)
                return True, f"Deleted {deleted_count} documents"
            else:
                return False, f"Delete failed: {response.status_code} - {response.text}"
                
        except Exception as e:
            return False, f"Error deleting documents: {str(e)}"

class DataStaxEmbeddingManager:
    """Manage embeddings for DataStax integration"""
    
    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.client = None
        self._setup_openai_client()
    
    def _setup_openai_client(self):
        """Setup OpenAI client for embeddings"""
        try:
            if self.openai_api_key:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.openai_api_key)
        except ImportError:
            logger.error("OpenAI package not installed")
        except Exception as e:
            logger.error(f"Error setting up OpenAI client: {e}")
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using OpenAI"""
        try:
            if not self.client:
                return None
            
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts"""
        embeddings = []
        
        for text in texts:
            embedding = self.generate_embedding(text)
            embeddings.append(embedding)
        
        return embeddings

def get_datastax_vector_store():
    """Factory function to get DataStax vector store"""
    return DataStaxVectorStore()

def get_embedding_manager():
    """Factory function to get embedding manager"""
    return DataStaxEmbeddingManager()
