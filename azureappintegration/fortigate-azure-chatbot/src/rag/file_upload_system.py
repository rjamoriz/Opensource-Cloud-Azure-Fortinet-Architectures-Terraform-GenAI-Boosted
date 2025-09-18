"""
File Upload and Document Management System for RAG Agent
"""

import logging
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from datetime import datetime
import json
import uuid

from .config import get_rag_config
from .document_processor import get_document_processor, get_text_chunker
from .embedding_manager import get_embedding_manager
from .vector_store_manager import get_vector_store_manager

logger = logging.getLogger(__name__)

class FileUploadManager:
    """Manages file upload and document processing workflow"""
    
    def __init__(self):
        self.config = get_rag_config()
        self.document_processor = get_document_processor()
        self.text_chunker = get_text_chunker()
        self.embedding_manager = get_embedding_manager()
        self.vector_store_manager = get_vector_store_manager()
    
    def create_upload_interface(self) -> Tuple[List[Any], Dict[str, Any]]:
        """Create the file upload interface"""
        st.subheader("📄 Document Upload")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=self.document_processor.supported_extensions,
            accept_multiple_files=True,
            help=f"Supported formats: {', '.join(self.document_processor.supported_extensions)}"
        )
        
        # Upload options
        col1, col2 = st.columns(2)
        
        with col1:
            overwrite_existing = st.checkbox(
                "Overwrite existing documents",
                value=False,
                help="Replace documents with the same filename if they already exist"
            )
        
        with col2:
            auto_process = st.checkbox(
                "Auto-process uploaded files",
                value=True,
                help="Automatically chunk and embed uploaded documents"
            )
        
        # Advanced options
        with st.expander("Advanced Upload Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                custom_metadata = st.text_area(
                    "Custom Metadata (JSON)",
                    value="{}",
                    help="Add custom metadata in JSON format"
                )
            
            with col2:
                chunk_strategy = st.selectbox(
                    "Chunking Strategy",
                    options=["recursive", "fixed", "semantic"],
                    index=0,
                    help="Strategy for splitting documents into chunks"
                )
                
                chunk_size = st.slider(
                    "Chunk Size",
                    min_value=100,
                    max_value=2000,
                    value=self.config.chunk_size,
                    help="Maximum size of text chunks"
                )
                
                chunk_overlap = st.slider(
                    "Chunk Overlap",
                    min_value=0,
                    max_value=200,
                    value=self.config.chunk_overlap,
                    help="Overlap between consecutive chunks"
                )
        
        # Parse custom metadata
        try:
            custom_metadata_dict = json.loads(custom_metadata) if custom_metadata.strip() else {}
        except json.JSONDecodeError:
            st.error("Invalid JSON in custom metadata")
            custom_metadata_dict = {}
        
        upload_options = {
            "overwrite_existing": overwrite_existing,
            "auto_process": auto_process,
            "custom_metadata": custom_metadata_dict,
            "chunk_strategy": chunk_strategy,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        }
        
        return uploaded_files, upload_options
    
    def process_uploaded_files(self, uploaded_files: List[Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Process uploaded files and add them to the knowledge base"""
        if not uploaded_files:
            return {"success": False, "message": "No files to process"}
        
        results = {
            "success": True,
            "processed_files": [],
            "failed_files": [],
            "total_chunks": 0,
            "processing_time": 0
        }
        
        start_time = datetime.now()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            for i, uploaded_file in enumerate(uploaded_files):
                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"Processing {uploaded_file.name}...")
                
                try:
                    # Validate file
                    validation_result = self.document_processor.validate_file(uploaded_file)
                    
                    if not validation_result["valid"]:
                        results["failed_files"].append({
                            "filename": uploaded_file.name,
                            "errors": validation_result["errors"]
                        })
                        continue
                    
                    # Check if file already exists
                    if not options["overwrite_existing"]:
                        existing_docs = self.vector_store_manager.get_documents_by_filename(uploaded_file.name)
                        if existing_docs:
                            results["failed_files"].append({
                                "filename": uploaded_file.name,
                                "errors": ["File already exists. Enable 'Overwrite existing' to replace it."]
                            })
                            continue
                    
                    # Extract text
                    text_content, metadata = self.document_processor.extract_text_from_file(uploaded_file)
                    
                    # Add custom metadata
                    metadata.update(options["custom_metadata"])
                    metadata["upload_timestamp"] = datetime.now().isoformat()
                    metadata["upload_id"] = str(uuid.uuid4())
                    
                    # Update chunking configuration temporarily
                    original_chunk_size = self.config.chunk_size
                    original_chunk_overlap = self.config.chunk_overlap
                    original_chunk_strategy = self.config.chunk_strategy
                    
                    self.config.chunk_size = options["chunk_size"]
                    self.config.chunk_overlap = options["chunk_overlap"]
                    self.config.chunk_strategy = options["chunk_strategy"]
                    
                    try:
                        # Chunk text
                        chunks = self.text_chunker.chunk_text(text_content, metadata)
                        
                        if options["auto_process"]:
                            # Generate embeddings
                            enriched_chunks = self.embedding_manager.embed_chunks(chunks)
                            
                            # Remove existing documents with same filename if overwriting
                            if options["overwrite_existing"]:
                                self.vector_store_manager.delete_documents_by_filename(uploaded_file.name)
                            
                            # Add to vector store
                            self.vector_store_manager.add_documents(enriched_chunks)
                            
                            results["processed_files"].append({
                                "filename": uploaded_file.name,
                                "chunks": len(enriched_chunks),
                                "text_length": len(text_content),
                                "metadata": metadata
                            })
                            
                            results["total_chunks"] += len(enriched_chunks)
                        else:
                            # Just store the chunks without embeddings for later processing
                            results["processed_files"].append({
                                "filename": uploaded_file.name,
                                "chunks": len(chunks),
                                "text_length": len(text_content),
                                "metadata": metadata,
                                "auto_processed": False
                            })
                    
                    finally:
                        # Restore original configuration
                        self.config.chunk_size = original_chunk_size
                        self.config.chunk_overlap = original_chunk_overlap
                        self.config.chunk_strategy = original_chunk_strategy
                
                except Exception as e:
                    logger.error(f"Failed to process {uploaded_file.name}: {e}")
                    results["failed_files"].append({
                        "filename": uploaded_file.name,
                        "errors": [str(e)]
                    })
            
            processing_time = (datetime.now() - start_time).total_seconds()
            results["processing_time"] = processing_time
            
            # Update progress
            progress_bar.progress(1.0)
            status_text.text("Processing complete!")
            
            # Show results
            self._display_processing_results(results)
            
        except Exception as e:
            logger.error(f"Failed to process uploaded files: {e}")
            results["success"] = False
            results["error"] = str(e)
            st.error(f"Error processing files: {e}")
        
        return results
    
    def _display_processing_results(self, results: Dict[str, Any]):
        """Display processing results to the user"""
        if results["success"]:
            if results["processed_files"]:
                st.success(f"✅ Successfully processed {len(results['processed_files'])} files")
                
                # Show processed files
                with st.expander("Processed Files Details"):
                    for file_info in results["processed_files"]:
                        st.write(f"**{file_info['filename']}**")
                        st.write(f"- Chunks: {file_info['chunks']}")
                        st.write(f"- Text length: {file_info['text_length']:,} characters")
                        if file_info.get("auto_processed", True):
                            st.write("- ✅ Auto-processed (embedded and indexed)")
                        else:
                            st.write("- ⏸️ Stored but not processed (manual processing required)")
                        st.write("---")
                
                if results["total_chunks"] > 0:
                    st.info(f"📊 Total chunks created: {results['total_chunks']}")
                
                st.info(f"⏱️ Processing time: {results['processing_time']:.2f} seconds")
            
            if results["failed_files"]:
                st.warning(f"⚠️ Failed to process {len(results['failed_files'])} files")
                
                with st.expander("Failed Files Details"):
                    for file_info in results["failed_files"]:
                        st.write(f"**{file_info['filename']}**")
                        for error in file_info["errors"]:
                            st.write(f"- ❌ {error}")
                        st.write("---")
        else:
            st.error(f"❌ Processing failed: {results.get('error', 'Unknown error')}")

class DocumentManager:
    """Manages existing documents in the knowledge base"""
    
    def __init__(self):
        self.vector_store_manager = get_vector_store_manager()
    
    def create_document_management_interface(self):
        """Create the document management interface"""
        st.subheader("📚 Document Management")
        
        # Get document statistics
        try:
            stats = self.vector_store_manager.get_stats()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Documents", stats.get("total_documents", 0))
            
            with col2:
                st.metric("Total Chunks", stats.get("total_chunks", 0))
            
            with col3:
                st.metric("Vector Store Health", "🟢 Healthy" if stats.get("healthy", False) else "🔴 Unhealthy")
            
        except Exception as e:
            st.error(f"Failed to load document statistics: {e}")
            return
        
        # Document list and management
        tab1, tab2, tab3 = st.tabs(["📋 Document List", "🔍 Search Documents", "🗑️ Manage Documents"])
        
        with tab1:
            self._show_document_list()
        
        with tab2:
            self._show_document_search()
        
        with tab3:
            self._show_document_management()
    
    def _show_document_list(self):
        """Show list of all documents"""
        try:
            # Get all documents (this might need pagination for large collections)
            documents = self.vector_store_manager.list_all_documents()
            
            if not documents:
                st.info("No documents found in the knowledge base.")
                return
            
            # Group documents by filename
            document_groups = {}
            for doc in documents:
                filename = doc.get("metadata", {}).get("filename", "Unknown")
                if filename not in document_groups:
                    document_groups[filename] = []
                document_groups[filename].append(doc)
            
            # Display document groups
            for filename, docs in document_groups.items():
                with st.expander(f"📄 {filename} ({len(docs)} chunks)"):
                    # Show document metadata
                    first_doc = docs[0]
                    metadata = first_doc.get("metadata", {})
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Metadata:**")
                        for key, value in metadata.items():
                            if key not in ["chunk_id", "chunk_length", "chunk_word_count"]:
                                st.write(f"- {key}: {value}")
                    
                    with col2:
                        st.write("**Statistics:**")
                        total_length = sum(doc.get("metadata", {}).get("chunk_length", 0) for doc in docs)
                        total_words = sum(doc.get("metadata", {}).get("chunk_word_count", 0) for doc in docs)
                        st.write(f"- Total chunks: {len(docs)}")
                        st.write(f"- Total length: {total_length:,} characters")
                        st.write(f"- Total words: {total_words:,}")
                    
                    # Show first few chunks
                    if st.button(f"Show chunks for {filename}", key=f"show_chunks_{filename}"):
                        for i, doc in enumerate(docs[:3]):  # Show first 3 chunks
                            st.write(f"**Chunk {i+1}:**")
                            st.write(doc.get("content", "")[:200] + "..." if len(doc.get("content", "")) > 200 else doc.get("content", ""))
                            st.write("---")
                        
                        if len(docs) > 3:
                            st.write(f"... and {len(docs) - 3} more chunks")
        
        except Exception as e:
            st.error(f"Failed to load document list: {e}")
    
    def _show_document_search(self):
        """Show document search interface"""
        search_query = st.text_input("Search documents:", placeholder="Enter search terms...")
        
        if search_query:
            try:
                # Get embedding for search query
                embedding_manager = get_embedding_manager()
                query_embedding = embedding_manager.embed_text(search_query)
                
                # Search for similar documents
                results = self.vector_store_manager.search_similar(
                    query_embedding=query_embedding,
                    k=10
                )
                
                if results:
                    st.write(f"Found {len(results)} relevant chunks:")
                    
                    for i, result in enumerate(results):
                        with st.expander(f"Result {i+1} - {result.get('metadata', {}).get('filename', 'Unknown')} (Score: {result.get('score', 0):.3f})"):
                            st.write("**Content:**")
                            st.write(result.get("content", ""))
                            
                            st.write("**Metadata:**")
                            metadata = result.get("metadata", {})
                            for key, value in metadata.items():
                                st.write(f"- {key}: {value}")
                else:
                    st.info("No relevant documents found.")
            
            except Exception as e:
                st.error(f"Search failed: {e}")
    
    def _show_document_management(self):
        """Show document management options"""
        st.write("**Bulk Operations:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Refresh Vector Store", help="Refresh the vector store connection"):
                try:
                    self.vector_store_manager.health_check()
                    st.success("Vector store refreshed successfully!")
                except Exception as e:
                    st.error(f"Failed to refresh vector store: {e}")
        
        with col2:
            if st.button("📊 Recalculate Statistics", help="Recalculate document statistics"):
                try:
                    stats = self.vector_store_manager.get_stats()
                    st.json(stats)
                except Exception as e:
                    st.error(f"Failed to calculate statistics: {e}")
        
        st.write("**Individual Document Operations:**")
        
        # Get list of filenames
        try:
            documents = self.vector_store_manager.list_all_documents()
            filenames = list(set(doc.get("metadata", {}).get("filename", "Unknown") for doc in documents))
            
            if filenames:
                selected_filename = st.selectbox("Select document to manage:", filenames)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button(f"🗑️ Delete {selected_filename}", help="Delete all chunks for this document"):
                        if st.session_state.get(f"confirm_delete_{selected_filename}", False):
                            try:
                                self.vector_store_manager.delete_documents_by_filename(selected_filename)
                                st.success(f"Deleted {selected_filename} successfully!")
                                st.session_state[f"confirm_delete_{selected_filename}"] = False
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to delete {selected_filename}: {e}")
                        else:
                            st.session_state[f"confirm_delete_{selected_filename}"] = True
                            st.warning("Click again to confirm deletion!")
                
                with col2:
                    if st.button(f"📋 Export {selected_filename}", help="Export document data"):
                        try:
                            doc_data = self.vector_store_manager.get_documents_by_filename(selected_filename)
                            if doc_data:
                                # Convert to downloadable format
                                export_data = {
                                    "filename": selected_filename,
                                    "export_timestamp": datetime.now().isoformat(),
                                    "chunks": doc_data
                                }
                                
                                json_str = json.dumps(export_data, indent=2)
                                st.download_button(
                                    label=f"Download {selected_filename}.json",
                                    data=json_str,
                                    file_name=f"{selected_filename}_export.json",
                                    mime="application/json"
                                )
                            else:
                                st.error("No data found for this document")
                        except Exception as e:
                            st.error(f"Failed to export {selected_filename}: {e}")
            else:
                st.info("No documents available for management.")
        
        except Exception as e:
            st.error(f"Failed to load documents for management: {e}")

# Global instances
_file_upload_manager = None
_document_manager = None

def get_file_upload_manager() -> FileUploadManager:
    """Get the global file upload manager instance"""
    global _file_upload_manager
    if _file_upload_manager is None:
        _file_upload_manager = FileUploadManager()
    return _file_upload_manager

def get_document_manager() -> DocumentManager:
    """Get the global document manager instance"""
    global _document_manager
    if _document_manager is None:
        _document_manager = DocumentManager()
    return _document_manager
