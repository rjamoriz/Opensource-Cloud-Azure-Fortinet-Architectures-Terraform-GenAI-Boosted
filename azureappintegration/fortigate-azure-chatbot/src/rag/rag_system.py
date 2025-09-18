"""
RAG System Core Implementation
FortiGate Azure Chatbot - Retrieval-Augmented Generation
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib

# Core dependencies
try:
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.document_loaders import TextLoader, PDFLoader, JSONLoader
    from langchain.schema import Document
    from langchain.chains import RetrievalQA
    from langchain.llms import OpenAI
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Create a fallback Document class if langchain is not available
    class Document:
        def __init__(self, page_content: str, metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}
    
    LANGCHAIN_AVAILABLE = False

# Graph database
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

# Vector store alternatives
try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    similarity_threshold: float = 0.8
    max_retrieved_docs: int = 5
    embedding_model: str = "text-embedding-ada-002"
    llm_model: str = "gpt-3.5-turbo"
    vector_store_path: str = "./data/vector_store"
    graph_db_uri: str = "bolt://localhost:7687"
    graph_db_user: str = "neo4j"
    graph_db_password: str = "password"

class DocumentProcessor:
    """Handles document processing and chunking"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def process_documents(self, file_paths: List[str]) -> List[Document]:
        """Process multiple documents into chunks"""
        documents = []
        
        for file_path in file_paths:
            try:
                # Load document based on file type
                if file_path.endswith('.pdf'):
                    loader = PDFLoader(file_path)
                elif file_path.endswith('.json'):
                    loader = JSONLoader(file_path)
                else:
                    loader = TextLoader(file_path)
                
                # Load and split document
                docs = loader.load()
                chunks = self.text_splitter.split_documents(docs)
                
                # Add metadata
                for chunk in chunks:
                    chunk.metadata.update({
                        'source_file': file_path,
                        'processed_at': datetime.now().isoformat(),
                        'chunk_id': self._generate_chunk_id(chunk.page_content)
                    })
                
                documents.extend(chunks)
                logger.info(f"Processed {file_path}: {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        return documents
    
    def _generate_chunk_id(self, content: str) -> str:
        """Generate unique ID for chunk"""
        return hashlib.md5(content.encode()).hexdigest()[:12]

class VectorStoreManager:
    """Manages vector store operations"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.embeddings = None
        self.vector_store = None
        
        if LANGCHAIN_AVAILABLE:
            self.embeddings = OpenAIEmbeddings(
                model=config.embedding_model,
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
    
    def initialize_vector_store(self):
        """Initialize or load vector store"""
        if not LANGCHAIN_AVAILABLE:
            logger.error("LangChain not available")
            return False
        
        try:
            # Create or load Chroma vector store
            self.vector_store = Chroma(
                persist_directory=self.config.vector_store_path,
                embedding_function=self.embeddings
            )
            logger.info("Vector store initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            return False
    
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to vector store"""
        if not self.vector_store:
            logger.error("Vector store not initialized")
            return False
        
        try:
            self.vector_store.add_documents(documents)
            self.vector_store.persist()
            logger.info(f"Added {len(documents)} documents to vector store")
            return True
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """Perform similarity search"""
        if not self.vector_store:
            return []
        
        k = k or self.config.max_retrieved_docs
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            # Filter by similarity threshold
            filtered_results = [
                doc for doc, score in results 
                if score >= self.config.similarity_threshold
            ]
            return filtered_results
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []

class GraphStoreManager:
    """Manages Neo4j graph database operations"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.driver = None
        
        if NEO4J_AVAILABLE:
            try:
                self.driver = GraphDatabase.driver(
                    config.graph_db_uri,
                    auth=(config.graph_db_user, config.graph_db_password)
                )
                logger.info("Neo4j driver initialized")
            except Exception as e:
                logger.error(f"Error connecting to Neo4j: {e}")
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
    
    def create_knowledge_graph(self, documents: List[Document]):
        """Create knowledge graph from documents"""
        if not self.driver:
            logger.error("Neo4j driver not available")
            return
        
        with self.driver.session() as session:
            for doc in documents:
                # Extract entities and relationships (simplified)
                entities = self._extract_entities(doc.page_content)
                relationships = self._extract_relationships(doc.page_content)
                
                # Create nodes and relationships
                self._create_nodes(session, entities, doc.metadata)
                self._create_relationships(session, relationships)
    
    def _extract_entities(self, text: str) -> List[Dict]:
        """Extract entities from text (simplified implementation)"""
        # This would use NER models in production
        azure_services = ['Azure VM', 'Azure VNET', 'Azure Load Balancer', 'Azure Storage']
        fortigate_features = ['FortiGate HA', 'FortiGate Firewall', 'FortiGate VPN']
        
        entities = []
        for service in azure_services:
            if service.lower() in text.lower():
                entities.append({
                    'name': service,
                    'type': 'AzureService',
                    'context': text[:200]  # First 200 chars for context
                })
        
        for feature in fortigate_features:
            if feature.lower() in text.lower():
                entities.append({
                    'name': feature,
                    'type': 'FortiGateFeature',
                    'context': text[:200]
                })
        
        return entities
    
    def _extract_relationships(self, text: str) -> List[Dict]:
        """Extract relationships from text (simplified implementation)"""
        relationships = []
        
        # Simple pattern matching for relationships
        if 'integrates with' in text.lower():
            relationships.append({
                'type': 'INTEGRATES_WITH',
                'context': text
            })
        
        if 'depends on' in text.lower():
            relationships.append({
                'type': 'DEPENDS_ON',
                'context': text
            })
        
        return relationships
    
    def _create_nodes(self, session, entities: List[Dict], metadata: Dict):
        """Create nodes in Neo4j"""
        for entity in entities:
            query = f"""
            MERGE (n:{entity['type']} {{name: $name}})
            SET n.context = $context,
                n.source = $source,
                n.updated_at = datetime()
            """
            session.run(query, 
                       name=entity['name'],
                       context=entity['context'],
                       source=metadata.get('source_file', 'unknown'))
    
    def _create_relationships(self, session, relationships: List[Dict]):
        """Create relationships in Neo4j"""
        for rel in relationships:
            # Simplified relationship creation
            query = f"""
            MATCH (a), (b)
            WHERE a.context CONTAINS $context AND b.context CONTAINS $context
            MERGE (a)-[r:{rel['type']}]->(b)
            SET r.context = $context
            """
            session.run(query, context=rel['context'])
    
    def query_graph(self, query: str) -> List[Dict]:
        """Query the knowledge graph"""
        if not self.driver:
            return []
        
        # Convert natural language to Cypher (simplified)
        cypher_query = self._natural_to_cypher(query)
        
        with self.driver.session() as session:
            result = session.run(cypher_query)
            return [record.data() for record in result]
    
    def _natural_to_cypher(self, query: str) -> str:
        """Convert natural language to Cypher query (simplified)"""
        # This would use more sophisticated NLP in production
        if 'azure' in query.lower() and 'fortigate' in query.lower():
            return """
            MATCH (a:AzureService)-[r:INTEGRATES_WITH]-(f:FortiGateFeature)
            RETURN a.name, r.type, f.name, a.context
            LIMIT 10
            """
        elif 'azure' in query.lower():
            return """
            MATCH (a:AzureService)
            RETURN a.name, a.context
            LIMIT 10
            """
        else:
            return """
            MATCH (n)
            RETURN n.name, n.context
            LIMIT 10
            """

class QueryRouter:
    """Routes queries to appropriate retrieval methods"""
    
    def __init__(self):
        self.query_patterns = {
            'relationship': ['how does', 'relationship', 'connects to', 'integrates with'],
            'semantic': ['what is', 'explain', 'describe', 'definition'],
            'procedural': ['how to', 'steps', 'configure', 'setup'],
            'troubleshooting': ['error', 'problem', 'issue', 'troubleshoot', 'fix']
        }
    
    def classify_query(self, query: str) -> str:
        """Classify query type"""
        query_lower = query.lower()
        
        for query_type, patterns in self.query_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return query_type
        
        return 'semantic'  # Default to semantic search

class FortiGateAzureRAGSystem:
    """Main RAG system orchestrator"""
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.document_processor = DocumentProcessor(self.config)
        self.vector_store = VectorStoreManager(self.config)
        self.graph_store = GraphStoreManager(self.config)
        self.query_router = QueryRouter()
        self.qa_chain = None
        
        # Initialize components
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize RAG system components"""
        logger.info("Initializing RAG system...")
        
        # Initialize vector store
        if self.vector_store.initialize_vector_store():
            logger.info("Vector store ready")
        
        # Initialize QA chain
        if LANGCHAIN_AVAILABLE and self.vector_store.vector_store:
            try:
                llm = OpenAI(
                    model_name=self.config.llm_model,
                    openai_api_key=os.getenv('OPENAI_API_KEY')
                )
                
                # Custom prompt for FortiGate Azure context
                prompt_template = """
                You are a specialized FortiGate Azure integration assistant. Use the following context to answer questions about FortiGate deployments on Azure.
                
                Context: {context}
                
                Question: {question}
                
                Provide detailed, accurate answers based on the context. If the context doesn't contain enough information, say so and provide general guidance.
                
                Answer:
                """
                
                prompt = PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "question"]
                )
                
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=self.vector_store.vector_store.as_retriever(),
                    chain_type_kwargs={"prompt": prompt}
                )
                
                logger.info("QA chain initialized")
            except Exception as e:
                logger.error(f"Error initializing QA chain: {e}")
    
    def ingest_documents(self, file_paths: List[str]) -> bool:
        """Ingest documents into the RAG system"""
        logger.info(f"Ingesting {len(file_paths)} documents...")
        
        # Process documents
        documents = self.document_processor.process_documents(file_paths)
        if not documents:
            logger.error("No documents processed")
            return False
        
        # Add to vector store
        vector_success = self.vector_store.add_documents(documents)
        
        # Add to graph store
        try:
            self.graph_store.create_knowledge_graph(documents)
            graph_success = True
        except Exception as e:
            logger.error(f"Error creating knowledge graph: {e}")
            graph_success = False
        
        logger.info(f"Document ingestion complete. Vector: {vector_success}, Graph: {graph_success}")
        return vector_success or graph_success
    
    def query(self, question: str) -> Dict[str, Any]:
        """Process a query using the RAG system"""
        start_time = datetime.now()
        
        # Classify query
        query_type = self.query_router.classify_query(question)
        
        # Retrieve context based on query type
        if query_type == 'relationship' and self.graph_store.driver:
            # Use graph retrieval for relationship queries
            graph_results = self.graph_store.query_graph(question)
            context = self._format_graph_results(graph_results)
            retrieval_method = 'graph'
        else:
            # Use vector similarity search
            similar_docs = self.vector_store.similarity_search(question)
            context = self._format_vector_results(similar_docs)
            retrieval_method = 'vector'
        
        # Generate response
        if self.qa_chain and context:
            try:
                response = self.qa_chain.run(question)
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                response = f"I found relevant information but encountered an error generating the response. Context: {context[:500]}..."
        else:
            response = f"Based on the available information: {context[:1000]}..."
        
        # Calculate response time
        response_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'question': question,
            'answer': response,
            'query_type': query_type,
            'retrieval_method': retrieval_method,
            'context': context,
            'response_time': response_time,
            'timestamp': datetime.now().isoformat()
        }
    
    def _format_graph_results(self, results: List[Dict]) -> str:
        """Format graph query results"""
        if not results:
            return "No relevant relationships found."
        
        formatted = "Relevant relationships:\n"
        for result in results[:3]:  # Limit to top 3 results
            formatted += f"- {result}\n"
        
        return formatted
    
    def _format_vector_results(self, documents: List[Document]) -> str:
        """Format vector search results"""
        if not documents:
            return "No relevant documents found."
        
        context = ""
        for i, doc in enumerate(documents[:3]):  # Limit to top 3 results
            context += f"Source {i+1}: {doc.page_content[:300]}...\n\n"
        
        return context
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = {
            'vector_store_initialized': self.vector_store.vector_store is not None,
            'graph_store_initialized': self.graph_store.driver is not None,
            'qa_chain_initialized': self.qa_chain is not None,
            'langchain_available': LANGCHAIN_AVAILABLE,
            'neo4j_available': NEO4J_AVAILABLE,
            'config': {
                'chunk_size': self.config.chunk_size,
                'similarity_threshold': self.config.similarity_threshold,
                'max_retrieved_docs': self.config.max_retrieved_docs
            }
        }
        
        return stats
    
    def close(self):
        """Clean up resources"""
        if self.graph_store:
            self.graph_store.close()

# Example usage and testing
if __name__ == "__main__":
    # Initialize RAG system
    rag_system = FortiGateAzureRAGSystem()
    
    # Test query
    test_query = "How do I configure FortiGate HA on Azure?"
    result = rag_system.query(test_query)
    
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Query Type: {result['query_type']}")
    print(f"Response Time: {result['response_time']:.2f}s")
    
    # Get system stats
    stats = rag_system.get_system_stats()
    print(f"System Stats: {json.dumps(stats, indent=2)}")
    
    # Clean up
    rag_system.close()
