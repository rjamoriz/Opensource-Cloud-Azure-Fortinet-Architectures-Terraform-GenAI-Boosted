"""
Query Processing and Response Generation for RAG Agent
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Iterator
import asyncio
from datetime import datetime

# LangChain imports
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from .config import get_rag_config
from .vector_store_manager import get_vector_store_manager
from .embedding_manager import get_embedding_manager
from .langchain_agent import get_rag_agent

logger = logging.getLogger(__name__)

class QueryProcessor:
    """Handles query processing and context retrieval"""
    
    def __init__(self):
        self.config = get_rag_config()
        self.vector_store_manager = get_vector_store_manager()
        self.embedding_manager = get_embedding_manager()
    
    def process_query(self, query: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a user query and retrieve relevant context"""
        try:
            start_time = datetime.now()
            
            # Generate query embedding
            query_embedding = self.embedding_manager.embed_text(query)
            
            # Search for relevant chunks
            search_results = self.vector_store_manager.search_similar(
                query_embedding=query_embedding,
                k=self.config.retrieval_k,
                filters=filters
            )
            
            # Process and rank results
            processed_results = self._process_search_results(search_results, query)
            
            # Generate context
            context = self._generate_context(processed_results)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "query": query,
                "context": context,
                "sources": processed_results,
                "processing_time": processing_time,
                "num_sources": len(processed_results)
            }
            
        except Exception as e:
            logger.error(f"Failed to process query: {e}")
            raise
    
    def _process_search_results(self, search_results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Process and rank search results"""
        try:
            processed_results = []
            
            for result in search_results:
                # Extract relevant information
                chunk_data = {
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.0),
                    "metadata": result.get("metadata", {}),
                    "source": result.get("metadata", {}).get("filename", "Unknown"),
                    "chunk_id": result.get("id", ""),
                }
                
                # Add relevance score (could be enhanced with re-ranking)
                chunk_data["relevance_score"] = self._calculate_relevance_score(
                    chunk_data["content"], 
                    query, 
                    chunk_data["score"]
                )
                
                processed_results.append(chunk_data)
            
            # Sort by relevance score
            processed_results.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Failed to process search results: {e}")
            return []
    
    def _calculate_relevance_score(self, content: str, query: str, similarity_score: float) -> float:
        """Calculate relevance score combining similarity and keyword matching"""
        try:
            # Start with similarity score
            relevance_score = similarity_score
            
            # Add keyword matching bonus
            query_words = set(query.lower().split())
            content_words = set(content.lower().split())
            keyword_overlap = len(query_words.intersection(content_words)) / len(query_words)
            relevance_score += keyword_overlap * 0.1
            
            # Add length penalty for very short or very long chunks
            content_length = len(content)
            if content_length < 50:
                relevance_score *= 0.8  # Penalty for very short chunks
            elif content_length > 2000:
                relevance_score *= 0.9  # Small penalty for very long chunks
            
            return relevance_score
            
        except Exception as e:
            logger.warning(f"Failed to calculate relevance score: {e}")
            return similarity_score
    
    def _generate_context(self, processed_results: List[Dict[str, Any]]) -> str:
        """Generate context string from processed results"""
        try:
            if not processed_results:
                return "No relevant context found."
            
            context_parts = []
            for i, result in enumerate(processed_results[:self.config.retrieval_k]):
                source = result["source"]
                content = result["content"]
                score = result["relevance_score"]
                
                context_part = f"[Source {i+1}: {source} (Relevance: {score:.3f})]\n{content}"
                context_parts.append(context_part)
            
            return "\n\n" + "="*50 + "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Failed to generate context: {e}")
            return "Error generating context."

class ResponseGenerator:
    """Handles response generation using retrieved context"""
    
    def __init__(self):
        self.config = get_rag_config()
        self.query_processor = QueryProcessor()
        self.rag_agent = get_rag_agent()
        
        # Initialize prompt template
        self.prompt_template = self._create_prompt_template()
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the RAG prompt template"""
        try:
            system_message = """You are an expert assistant specializing in cloud infrastructure, virtualization, and network security. 
            You help users with Azure, GCP, AWS, and multi-cloud deployments, particularly with FortiGate and Fortinet solutions.

            Use the provided context to answer questions accurately and comprehensively. If the context doesn't contain 
            enough information to fully answer the question, acknowledge this and provide what information you can.

            Context Guidelines:
            - Prioritize information from the provided context
            - If multiple sources provide different information, acknowledge the discrepancy
            - Be specific about which source supports your statements when relevant
            - If no relevant context is provided, say so and provide general knowledge if helpful

            Format your responses clearly with:
            - Direct answer to the question
            - Supporting details from the context
            - Practical recommendations when applicable
            - Source references when relevant
            """
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", """Based on the following context, please answer the question:

Context:
{context}

Question: {question}

Please provide a comprehensive answer based on the context provided.""")
            ])
            
            return prompt
            
        except Exception as e:
            logger.error(f"Failed to create prompt template: {e}")
            raise
    
    def generate_response(self, query: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a complete response to a user query"""
        try:
            start_time = datetime.now()
            
            # Process query and get context
            query_result = self.query_processor.process_query(query, filters)
            
            # Get chat history from agent
            chat_history = self.rag_agent.get_conversation_history()
            
            # Prepare prompt variables
            prompt_variables = {
                "context": query_result["context"],
                "question": query,
                "chat_history": chat_history
            }
            
            # Generate response using the agent's LLM
            llm = self.rag_agent.llm
            chain = self.prompt_template | llm | StrOutputParser()
            
            response = chain.invoke(prompt_variables)
            
            # Update conversation memory
            self.rag_agent.memory.chat_memory.add_user_message(query)
            self.rag_agent.memory.chat_memory.add_ai_message(response)
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "query": query,
                "response": response,
                "context": query_result["context"],
                "sources": query_result["sources"],
                "metadata": {
                    "processing_time": query_result["processing_time"],
                    "generation_time": generation_time,
                    "total_time": generation_time,
                    "num_sources": query_result["num_sources"],
                    "filters_applied": filters or {}
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise
    
    def generate_streaming_response(self, query: str, filters: Optional[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
        """Generate a streaming response to a user query"""
        try:
            # Process query and get context
            query_result = self.query_processor.process_query(query, filters)
            
            # Yield initial metadata
            yield {
                "type": "metadata",
                "data": {
                    "num_sources": query_result["num_sources"],
                    "processing_time": query_result["processing_time"]
                }
            }
            
            # Yield sources
            yield {
                "type": "sources",
                "data": query_result["sources"]
            }
            
            # Get chat history
            chat_history = self.rag_agent.get_conversation_history()
            
            # Prepare prompt variables
            prompt_variables = {
                "context": query_result["context"],
                "question": query,
                "chat_history": chat_history
            }
            
            # Generate streaming response
            llm = self.rag_agent.llm
            chain = self.prompt_template | llm
            
            response_chunks = []
            for chunk in chain.stream(prompt_variables):
                chunk_content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                response_chunks.append(chunk_content)
                
                yield {
                    "type": "response_chunk",
                    "data": chunk_content
                }
            
            # Combine full response and update memory
            full_response = "".join(response_chunks)
            self.rag_agent.memory.chat_memory.add_user_message(query)
            self.rag_agent.memory.chat_memory.add_ai_message(full_response)
            
            # Yield completion
            yield {
                "type": "complete",
                "data": {
                    "full_response": full_response
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to generate streaming response: {e}")
            yield {
                "type": "error",
                "data": {"error": str(e)}
            }
    
    async def generate_response_async(self, query: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate response asynchronously"""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self.generate_response,
                query,
                filters
            )
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate async response: {e}")
            raise
    
    def clear_conversation_history(self):
        """Clear the conversation history"""
        try:
            self.rag_agent.memory.clear()
            logger.info("Conversation history cleared")
        except Exception as e:
            logger.error(f"Failed to clear conversation history: {e}")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation"""
        try:
            history = self.rag_agent.get_conversation_history()
            
            return {
                "message_count": len(history),
                "user_messages": len([msg for msg in history if isinstance(msg, HumanMessage)]),
                "ai_messages": len([msg for msg in history if isinstance(msg, AIMessage)]),
                "last_messages": history[-4:] if len(history) > 4 else history
            }
            
        except Exception as e:
            logger.error(f"Failed to get conversation summary: {e}")
            return {}

# Global instances
_query_processor = None
_response_generator = None

def get_query_processor() -> QueryProcessor:
    """Get the global query processor instance"""
    global _query_processor
    if _query_processor is None:
        _query_processor = QueryProcessor()
    return _query_processor

def get_response_generator() -> ResponseGenerator:
    """Get the global response generator instance"""
    global _response_generator
    if _response_generator is None:
        _response_generator = ResponseGenerator()
    return _response_generator
