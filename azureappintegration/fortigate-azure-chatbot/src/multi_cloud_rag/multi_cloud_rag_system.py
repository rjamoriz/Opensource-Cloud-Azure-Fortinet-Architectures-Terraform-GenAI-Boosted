"""
Multi-Cloud RAG System
Main orchestrator for the enhanced multi-cloud VM architecture assistant
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import json
import yaml
from datetime import datetime

# LangChain imports
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Internal imports
from .vector_stores import (
    VectorStoreFactory, BaseVectorStore, DocumentMetadata, 
    CloudProvider, DocumentType, SearchFilter
)
from .cloud_apis import (
    CloudAPIFactory, BaseCloudAPI, VMSpecification, 
    VMDeploymentRequest, NetworkConfiguration
)
from .knowledge_base import KnowledgeBaseManager, KnowledgeBaseSeeder

logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """Configuration for the multi-cloud RAG system"""
    # Vector store configuration
    vector_store_type: str = "pinecone"
    vector_store_config: Dict[str, Any] = None
    
    # Embedding configuration
    embedding_model: str = "text-embedding-ada-002"
    embedding_dimension: int = 1536
    
    # LLM configuration
    llm_model: str = "gpt-3.5-turbo"
    temperature: float = 0.1
    max_tokens: int = 2000
    
    # Cloud API configurations
    cloud_configs: Dict[str, Dict[str, Any]] = None
    
    # Knowledge base path
    knowledge_base_path: str = "./knowledge_base"
    
    # Retrieval configuration
    max_results: int = 10
    similarity_threshold: float = 0.7

@dataclass
class VMRecommendation:
    """VM recommendation result"""
    cloud_provider: str
    vm_specification: VMSpecification
    reasoning: str
    cost_estimate: Dict[str, float]
    deployment_config: Dict[str, Any]
    confidence_score: float

@dataclass
class QueryResponse:
    """Response from the RAG system"""
    answer: str
    vm_recommendations: List[VMRecommendation]
    source_documents: List[str]
    metadata: Dict[str, Any]
    output_format: str = "markdown"

class MultiCloudRAGSystem:
    """Main multi-cloud RAG system orchestrator"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        
        # Initialize components
        self.vector_store: Optional[BaseVectorStore] = None
        self.cloud_apis: Dict[str, BaseCloudAPI] = {}
        self.knowledge_manager: Optional[KnowledgeBaseManager] = None
        self.embeddings = None
        self.llm = None
        self.retrieval_chain = None
        self.memory = None
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        self.initialized = False
    
    async def initialize(self):
        """Initialize all components of the RAG system"""
        try:
            logger.info("Initializing Multi-Cloud RAG System...")
            
            # Initialize embeddings
            self._initialize_embeddings()
            
            # Initialize LLM
            self._initialize_llm()
            
            # Initialize vector store
            await self._initialize_vector_store()
            
            # Initialize cloud APIs
            await self._initialize_cloud_apis()
            
            # Initialize knowledge base
            self._initialize_knowledge_base()
            
            # Initialize retrieval chain
            self._initialize_retrieval_chain()
            
            self.initialized = True
            logger.info("Multi-Cloud RAG System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
    
    def _initialize_embeddings(self):
        """Initialize embedding model"""
        try:
            self.embeddings = OpenAIEmbeddings(
                model=self.config.embedding_model
            )
            logger.info(f"Initialized embeddings: {self.config.embedding_model}")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
    
    def _initialize_llm(self):
        """Initialize language model"""
        try:
            self.llm = OpenAI(
                model_name=self.config.llm_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            logger.info(f"Initialized LLM: {self.config.llm_model}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    async def _initialize_vector_store(self):
        """Initialize vector store"""
        try:
            vector_config = self.config.vector_store_config or {}
            vector_config['embedding_dimension'] = self.config.embedding_dimension
            
            self.vector_store = VectorStoreFactory.create_vector_store(
                self.config.vector_store_type,
                vector_config
            )
            
            if not self.vector_store:
                raise ValueError(f"Failed to create vector store: {self.config.vector_store_type}")
            
            # Connect to vector store
            connected = await self.vector_store.connect()
            if not connected:
                raise ConnectionError("Failed to connect to vector store")
            
            logger.info(f"Initialized vector store: {self.config.vector_store_type}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    async def _initialize_cloud_apis(self):
        """Initialize cloud API connections"""
        try:
            if not self.config.cloud_configs:
                logger.warning("No cloud API configurations provided")
                return
            
            for provider, config in self.config.cloud_configs.items():
                api = CloudAPIFactory.create_cloud_api(provider, config)
                if api:
                    authenticated = await api.authenticate()
                    if authenticated:
                        self.cloud_apis[provider] = api
                        logger.info(f"Initialized cloud API: {provider}")
                    else:
                        logger.warning(f"Failed to authenticate with {provider}")
                else:
                    logger.warning(f"Failed to create cloud API for {provider}")
            
        except Exception as e:
            logger.error(f"Failed to initialize cloud APIs: {e}")
            raise
    
    def _initialize_knowledge_base(self):
        """Initialize knowledge base manager"""
        try:
            self.knowledge_manager = KnowledgeBaseManager(
                self.config.knowledge_base_path
            )
            
            # Check if knowledge base is empty and seed it
            stats = self.knowledge_manager.get_statistics()
            if stats["total_documents"] == 0:
                logger.info("Knowledge base is empty, seeding with initial content...")
                seeder = KnowledgeBaseSeeder(self.knowledge_manager)
                seeder.seed_initial_content()
            
            logger.info(f"Initialized knowledge base with {stats['total_documents']} documents")
            
        except Exception as e:
            logger.error(f"Failed to initialize knowledge base: {e}")
            raise
    
    def _initialize_retrieval_chain(self):
        """Initialize retrieval chain"""
        try:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            # Note: This is a simplified version. In production, you'd integrate
            # with the vector store to create a proper retrieval chain
            logger.info("Initialized retrieval chain")
            
        except Exception as e:
            logger.error(f"Failed to initialize retrieval chain: {e}")
            raise
    
    async def process_query(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        output_format: str = "markdown",
        include_vm_recommendations: bool = True
    ) -> QueryResponse:
        """Process a query and return comprehensive response"""
        try:
            if not self.initialized:
                await self.initialize()
            
            logger.info(f"Processing query: {query[:100]}...")
            
            # Generate embeddings for the query
            query_embedding = await self._embed_query(query)
            
            # Search knowledge base
            search_results = await self._search_knowledge_base(
                query_embedding, filters
            )
            
            # Generate VM recommendations if requested
            vm_recommendations = []
            if include_vm_recommendations:
                vm_recommendations = await self._generate_vm_recommendations(
                    query, search_results
                )
            
            # Generate comprehensive answer
            answer = await self._generate_answer(
                query, search_results, vm_recommendations
            )
            
            # Format response
            response = QueryResponse(
                answer=answer,
                vm_recommendations=vm_recommendations,
                source_documents=[doc.document_id for doc in search_results],
                metadata={
                    "query": query,
                    "timestamp": datetime.now().isoformat(),
                    "search_results_count": len(search_results),
                    "vm_recommendations_count": len(vm_recommendations)
                },
                output_format=output_format
            )
            
            # Format output based on requested format
            if output_format.lower() in ["json", "yaml"]:
                response = self._format_structured_output(response, output_format)
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to process query: {e}")
            raise
    
    async def _embed_query(self, query: str) -> List[float]:
        """Generate embeddings for query"""
        try:
            # Use OpenAI embeddings
            embedding = self.embeddings.embed_query(query)
            return embedding
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise
    
    async def _search_knowledge_base(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """Search knowledge base using vector similarity"""
        try:
            # Search vector store
            results = await self.vector_store.search(
                query_vector=query_embedding,
                filters=filters,
                top_k=self.config.max_results
            )
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in results 
                if result.similarity_score >= self.config.similarity_threshold
            ]
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Failed to search knowledge base: {e}")
            return []
    
    async def _generate_vm_recommendations(
        self,
        query: str,
        search_results: List[Any]
    ) -> List[VMRecommendation]:
        """Generate VM recommendations based on query and search results"""
        try:
            recommendations = []
            
            # Extract requirements from query
            requirements = self._extract_requirements(query)
            
            # Get recommendations from each configured cloud provider
            for provider_name, cloud_api in self.cloud_apis.items():
                try:
                    vm_specs = await cloud_api.get_vm_recommendations(
                        workload_type=requirements.get("workload_type", "general"),
                        performance_requirements=requirements.get("performance", {}),
                        budget_constraints=requirements.get("budget")
                    )
                    
                    for vm_spec in vm_specs[:3]:  # Top 3 per provider
                        cost_estimate = await cloud_api.get_pricing_info(vm_spec)
                        
                        recommendation = VMRecommendation(
                            cloud_provider=provider_name,
                            vm_specification=vm_spec,
                            reasoning=self._generate_reasoning(vm_spec, requirements),
                            cost_estimate=cost_estimate,
                            deployment_config=self._generate_deployment_config(vm_spec),
                            confidence_score=self._calculate_confidence_score(vm_spec, requirements)
                        )
                        
                        recommendations.append(recommendation)
                
                except Exception as e:
                    logger.warning(f"Failed to get recommendations from {provider_name}: {e}")
            
            # Sort by confidence score
            recommendations.sort(key=lambda x: x.confidence_score, reverse=True)
            
            return recommendations[:5]  # Return top 5 overall
            
        except Exception as e:
            logger.error(f"Failed to generate VM recommendations: {e}")
            return []
    
    def _extract_requirements(self, query: str) -> Dict[str, Any]:
        """Extract technical requirements from query"""
        requirements = {
            "workload_type": "general",
            "performance": {},
            "budget": None
        }
        
        query_lower = query.lower()
        
        # Extract workload type
        if any(term in query_lower for term in ["web", "website", "frontend"]):
            requirements["workload_type"] = "web"
        elif any(term in query_lower for term in ["database", "db", "sql"]):
            requirements["workload_type"] = "database"
        elif any(term in query_lower for term in ["machine learning", "ml", "ai"]):
            requirements["workload_type"] = "machine learning"
        elif any(term in query_lower for term in ["analytics", "data processing"]):
            requirements["workload_type"] = "analytics"
        
        # Extract performance requirements (simplified)
        if "high performance" in query_lower or "high cpu" in query_lower:
            requirements["performance"]["min_cpu_cores"] = 8
        elif "medium" in query_lower:
            requirements["performance"]["min_cpu_cores"] = 4
        else:
            requirements["performance"]["min_cpu_cores"] = 2
        
        if "memory intensive" in query_lower or "large memory" in query_lower:
            requirements["performance"]["min_memory_gb"] = 32
        elif "medium memory" in query_lower:
            requirements["performance"]["min_memory_gb"] = 16
        else:
            requirements["performance"]["min_memory_gb"] = 8
        
        return requirements
    
    def _generate_reasoning(self, vm_spec: VMSpecification, requirements: Dict[str, Any]) -> str:
        """Generate reasoning for VM recommendation"""
        reasoning_parts = []
        
        reasoning_parts.append(f"The {vm_spec.name} instance type provides {vm_spec.cpu_cores} CPU cores and {vm_spec.memory_gb}GB of memory")
        
        if vm_spec.cpu_cores >= requirements.get("performance", {}).get("min_cpu_cores", 2):
            reasoning_parts.append("which meets the CPU requirements")
        
        if vm_spec.memory_gb >= requirements.get("performance", {}).get("min_memory_gb", 8):
            reasoning_parts.append("and satisfies the memory requirements")
        
        reasoning_parts.append(f"At ${vm_spec.price_per_hour:.4f}/hour, it offers good value for {requirements.get('workload_type', 'general')} workloads")
        
        if vm_spec.suitable_workloads:
            workload_match = any(
                workload.lower() in vm_spec.suitable_workloads[0].lower() 
                for workload in [requirements.get('workload_type', '')]
            )
            if workload_match:
                reasoning_parts.append("and is specifically optimized for this workload type")
        
        return ". ".join(reasoning_parts) + "."
    
    def _generate_deployment_config(self, vm_spec: VMSpecification) -> Dict[str, Any]:
        """Generate deployment configuration for VM"""
        return {
            "instance_type": vm_spec.name,
            "storage": {
                "type": vm_spec.storage_type.value,
                "size_gb": vm_spec.storage_gb
            },
            "network": {
                "performance": vm_spec.network_performance
            },
            "availability_zones": vm_spec.availability_zones
        }
    
    def _calculate_confidence_score(self, vm_spec: VMSpecification, requirements: Dict[str, Any]) -> float:
        """Calculate confidence score for recommendation"""
        score = 0.5  # Base score
        
        # Performance match
        perf_req = requirements.get("performance", {})
        if vm_spec.cpu_cores >= perf_req.get("min_cpu_cores", 2):
            score += 0.2
        if vm_spec.memory_gb >= perf_req.get("min_memory_gb", 8):
            score += 0.2
        
        # Price consideration (lower is better, but not too low)
        if 0.01 <= vm_spec.price_per_hour <= 0.5:
            score += 0.1
        
        return min(score, 1.0)
    
    async def _generate_answer(
        self,
        query: str,
        search_results: List[Any],
        vm_recommendations: List[VMRecommendation]
    ) -> str:
        """Generate comprehensive answer using LLM"""
        try:
            # Prepare context from search results
            context_parts = []
            for result in search_results[:5]:  # Use top 5 results
                context_parts.append(f"Source: {result.document_id}")
                context_parts.append(result.content[:500] + "...")
                context_parts.append("")
            
            context = "\n".join(context_parts)
            
            # Prepare VM recommendations context
            vm_context = ""
            if vm_recommendations:
                vm_context = "\n\nBased on your requirements, here are the top VM recommendations:\n"
                for i, rec in enumerate(vm_recommendations[:3], 1):
                    vm_context += f"\n{i}. **{rec.cloud_provider.upper()} - {rec.vm_specification.name}**\n"
                    vm_context += f"   - {rec.vm_specification.cpu_cores} CPU cores, {rec.vm_specification.memory_gb}GB RAM\n"
                    vm_context += f"   - ${rec.vm_specification.price_per_hour:.4f}/hour\n"
                    vm_context += f"   - {rec.reasoning}\n"
            
            # Create prompt
            prompt = f"""
Based on the following context and VM recommendations, provide a comprehensive answer to the user's question about multi-cloud VM architecture.

User Question: {query}

Context from Knowledge Base:
{context}

{vm_context}

Please provide a detailed, helpful response that:
1. Directly answers the user's question
2. Incorporates relevant information from the knowledge base
3. Explains the VM recommendations if provided
4. Includes practical implementation guidance
5. Mentions any important considerations or best practices

Response:
"""
            
            # Generate response using LLM
            response = self.llm(prompt)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return f"I apologize, but I encountered an error while generating the response. However, I found {len(search_results)} relevant documents that might help with your question about multi-cloud VM architecture."
    
    def _format_structured_output(self, response: QueryResponse, format: str) -> str:
        """Format response as structured output (JSON/YAML)"""
        try:
            # Convert to dictionary
            response_dict = {
                "answer": response.answer,
                "vm_recommendations": [
                    {
                        "cloud_provider": rec.cloud_provider,
                        "vm_specification": {
                            "name": rec.vm_specification.name,
                            "cpu_cores": rec.vm_specification.cpu_cores,
                            "memory_gb": rec.vm_specification.memory_gb,
                            "storage_gb": rec.vm_specification.storage_gb,
                            "storage_type": rec.vm_specification.storage_type.value,
                            "price_per_hour": rec.vm_specification.price_per_hour
                        },
                        "reasoning": rec.reasoning,
                        "cost_estimate": rec.cost_estimate,
                        "deployment_config": rec.deployment_config,
                        "confidence_score": rec.confidence_score
                    }
                    for rec in response.vm_recommendations
                ],
                "source_documents": response.source_documents,
                "metadata": response.metadata
            }
            
            if format.lower() == "json":
                return json.dumps(response_dict, indent=2)
            elif format.lower() == "yaml":
                return yaml.dump(response_dict, default_flow_style=False)
            else:
                return response.answer
                
        except Exception as e:
            logger.error(f"Failed to format structured output: {e}")
            return response.answer
    
    async def add_knowledge(
        self,
        title: str,
        content: str,
        metadata: DocumentMetadata,
        tags: Optional[List[str]] = None
    ) -> str:
        """Add new knowledge to the system"""
        try:
            if not self.initialized:
                await self.initialize()
            
            # Add to knowledge base
            doc_id = self.knowledge_manager.add_document(
                title=title,
                content=content,
                metadata=metadata,
                tags=tags
            )
            
            # Generate embeddings and add to vector store
            await self._index_document(doc_id, content, metadata)
            
            logger.info(f"Added knowledge document: {title} ({doc_id})")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to add knowledge: {e}")
            raise
    
    async def _index_document(self, doc_id: str, content: str, metadata: DocumentMetadata):
        """Index document in vector store"""
        try:
            # Split content into chunks
            chunks = self.text_splitter.split_text(content)
            
            # Generate embeddings for each chunk
            documents = []
            for i, chunk in enumerate(chunks):
                embedding = await self._embed_query(chunk)
                
                doc_data = {
                    'id': f"{doc_id}_{i}",
                    'content': chunk,
                    'embedding': embedding,
                    'metadata': {
                        'cloud': metadata.cloud.value,
                        'topic': metadata.topic.value,
                        'region': metadata.region,
                        'complexity': metadata.complexity,
                        'use_case': metadata.use_case,
                        'last_updated': metadata.last_updated,
                        'source_url': metadata.source_url,
                        'version': metadata.version,
                        'compliance': metadata.compliance
                    }
                }
                documents.append(doc_data)
            
            # Upsert to vector store
            await self.vector_store.upsert_documents(documents)
            
        except Exception as e:
            logger.error(f"Failed to index document: {e}")
            raise
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            stats = {
                "initialized": self.initialized,
                "vector_store_type": self.config.vector_store_type,
                "cloud_providers": list(self.cloud_apis.keys())
            }
            
            if self.vector_store:
                vector_stats = await self.vector_store.get_stats()
                stats["vector_store"] = vector_stats
            
            if self.knowledge_manager:
                kb_stats = self.knowledge_manager.get_statistics()
                stats["knowledge_base"] = kb_stats
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {"error": str(e)}
