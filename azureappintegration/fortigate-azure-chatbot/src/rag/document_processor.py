"""
Document Processing Pipeline for RAG Agent
Handles file upload, parsing, and text processing
"""

import os
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import streamlit as st

# Document processing libraries
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from .config import get_rag_config

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document processing and text extraction"""
    
    def __init__(self):
        self.config = get_rag_config()
        self.supported_extensions = self._get_supported_extensions()
    
    def _get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions based on available libraries"""
        extensions = ['.txt', '.md']  # Always available
        
        if PDF_AVAILABLE:
            extensions.append('.pdf')
        if DOCX_AVAILABLE:
            extensions.extend(['.docx'])
        if PANDAS_AVAILABLE:
            extensions.extend(['.csv', '.xlsx'])
        
        return extensions
    
    def is_supported_file(self, filename: str) -> bool:
        """Check if file type is supported"""
        file_ext = Path(filename).suffix.lower()
        return file_ext in self.supported_extensions
    
    def validate_file(self, uploaded_file) -> Dict[str, Any]:
        """Validate uploaded file"""
        validation_result = {
            "valid": False,
            "filename": uploaded_file.name,
            "size_mb": uploaded_file.size / (1024 * 1024),
            "extension": Path(uploaded_file.name).suffix.lower(),
            "errors": []
        }
        
        # Check file size
        if validation_result["size_mb"] > self.config.max_file_size_mb:
            validation_result["errors"].append(f"File size ({validation_result['size_mb']:.2f} MB) exceeds limit ({self.config.max_file_size_mb} MB)")
        
        # Check file extension
        if not self.is_supported_file(uploaded_file.name):
            validation_result["errors"].append(f"File type '{validation_result['extension']}' not supported. Supported types: {', '.join(self.supported_extensions)}")
        
        # Check file is not empty
        if uploaded_file.size == 0:
            validation_result["errors"].append("File is empty")
        
        validation_result["valid"] = len(validation_result["errors"]) == 0
        return validation_result
    
    def extract_text_from_file(self, uploaded_file) -> Tuple[str, Dict[str, Any]]:
        """Extract text content from uploaded file"""
        try:
            file_ext = Path(uploaded_file.name).suffix.lower()
            
            # Generate file metadata
            file_content = uploaded_file.read()
            file_hash = hashlib.md5(file_content).hexdigest()
            
            metadata = {
                "filename": uploaded_file.name,
                "file_extension": file_ext,
                "file_size": len(file_content),
                "file_hash": file_hash,
                "file_type": uploaded_file.type if hasattr(uploaded_file, 'type') else 'unknown'
            }
            
            # Reset file pointer
            uploaded_file.seek(0)
            
            # Extract text based on file type
            if file_ext == '.txt':
                text = self._extract_text_from_txt(uploaded_file)
            elif file_ext == '.md':
                text = self._extract_text_from_markdown(uploaded_file)
            elif file_ext == '.pdf' and PDF_AVAILABLE:
                text = self._extract_text_from_pdf(uploaded_file)
            elif file_ext == '.docx' and DOCX_AVAILABLE:
                text = self._extract_text_from_docx(uploaded_file)
            elif file_ext == '.csv' and PANDAS_AVAILABLE:
                text = self._extract_text_from_csv(uploaded_file)
            elif file_ext == '.xlsx' and PANDAS_AVAILABLE:
                text = self._extract_text_from_excel(uploaded_file)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            # Add extraction metadata
            metadata.update({
                "text_length": len(text),
                "word_count": len(text.split()),
                "extraction_method": f"extract_from_{file_ext[1:]}"
            })
            
            logger.info(f"Extracted {len(text)} characters from {uploaded_file.name}")
            return text, metadata
            
        except Exception as e:
            logger.error(f"Failed to extract text from {uploaded_file.name}: {e}")
            raise
    
    def _extract_text_from_txt(self, file) -> str:
        """Extract text from TXT file"""
        try:
            content = file.read().decode('utf-8')
            return content
        except UnicodeDecodeError:
            # Try with different encoding
            file.seek(0)
            content = file.read().decode('latin-1')
            return content
    
    def _extract_text_from_markdown(self, file) -> str:
        """Extract text from Markdown file"""
        return self._extract_text_from_txt(file)  # Same as TXT for now
    
    def _extract_text_from_pdf(self, file) -> str:
        """Extract text from PDF file"""
        if not PDF_AVAILABLE:
            raise ImportError("PyPDF2 not available. Install with: pip install PyPDF2")
        
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text_content = []
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(f"--- Page {page_num + 1} ---\n{page_text}")
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                    continue
            
            return "\n\n".join(text_content)
            
        except Exception as e:
            logger.error(f"Failed to extract PDF text: {e}")
            raise
    
    def _extract_text_from_docx(self, file) -> str:
        """Extract text from DOCX file"""
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx not available. Install with: pip install python-docx")
        
        try:
            doc = DocxDocument(file)
            text_content = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content.append(paragraph.text)
            
            return "\n\n".join(text_content)
            
        except Exception as e:
            logger.error(f"Failed to extract DOCX text: {e}")
            raise
    
    def _extract_text_from_csv(self, file) -> str:
        """Extract text from CSV file"""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas not available. Install with: pip install pandas")
        
        try:
            df = pd.read_csv(file)
            
            # Convert DataFrame to text representation
            text_content = [f"CSV Data with {len(df)} rows and {len(df.columns)} columns\n"]
            text_content.append(f"Columns: {', '.join(df.columns)}\n")
            
            # Add sample data
            sample_size = min(10, len(df))
            text_content.append(f"Sample data (first {sample_size} rows):")
            text_content.append(df.head(sample_size).to_string())
            
            # Add summary statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                text_content.append("\nSummary Statistics:")
                text_content.append(df[numeric_cols].describe().to_string())
            
            return "\n\n".join(text_content)
            
        except Exception as e:
            logger.error(f"Failed to extract CSV text: {e}")
            raise
    
    def _extract_text_from_excel(self, file) -> str:
        """Extract text from Excel file"""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas and openpyxl not available. Install with: pip install pandas openpyxl")
        
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file)
            text_content = [f"Excel file with {len(excel_file.sheet_names)} sheets\n"]
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file, sheet_name=sheet_name)
                text_content.append(f"--- Sheet: {sheet_name} ---")
                text_content.append(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
                text_content.append(f"Columns: {', '.join(df.columns)}")
                
                # Add sample data
                sample_size = min(5, len(df))
                if sample_size > 0:
                    text_content.append(f"Sample data (first {sample_size} rows):")
                    text_content.append(df.head(sample_size).to_string())
                
                text_content.append("")  # Add spacing between sheets
            
            return "\n".join(text_content)
            
        except Exception as e:
            logger.error(f"Failed to extract Excel text: {e}")
            raise

class TextChunker:
    """Handles text chunking strategies"""
    
    def __init__(self):
        self.config = get_rag_config()
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk text using configured strategy"""
        try:
            if self.config.chunk_strategy == "recursive":
                return self._recursive_chunk(text, metadata)
            elif self.config.chunk_strategy == "fixed":
                return self._fixed_chunk(text, metadata)
            elif self.config.chunk_strategy == "semantic":
                return self._semantic_chunk(text, metadata)
            else:
                logger.warning(f"Unknown chunk strategy: {self.config.chunk_strategy}, using recursive")
                return self._recursive_chunk(text, metadata)
                
        except Exception as e:
            logger.error(f"Failed to chunk text: {e}")
            raise
    
    def _recursive_chunk(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recursive text chunking"""
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        
        # Simple recursive chunking by paragraphs, then sentences
        paragraphs = text.split('\n\n')
        current_chunk = ""
        chunk_id = 0
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) <= chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk.strip(), chunk_id, metadata))
                    chunk_id += 1
                
                # Handle overlap
                if overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-overlap:]
                    current_chunk = overlap_text + paragraph + "\n\n"
                else:
                    current_chunk = paragraph + "\n\n"
        
        # Add final chunk
        if current_chunk:
            chunks.append(self._create_chunk(current_chunk.strip(), chunk_id, metadata))
        
        return chunks
    
    def _fixed_chunk(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fixed-size text chunking"""
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk_text = text[i:i + chunk_size]
            if chunk_text.strip():
                chunks.append(self._create_chunk(chunk_text, i // (chunk_size - overlap), metadata))
        
        return chunks
    
    def _semantic_chunk(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Semantic text chunking (simplified version)"""
        # For now, use paragraph-based chunking as a semantic approach
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        
        for i, paragraph in enumerate(paragraphs):
            chunks.append(self._create_chunk(paragraph, i, metadata))
        
        return chunks
    
    def _create_chunk(self, text: str, chunk_id: int, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create a chunk dictionary"""
        chunk_metadata = metadata.copy()
        chunk_metadata.update({
            "chunk_id": chunk_id,
            "chunk_length": len(text),
            "chunk_word_count": len(text.split())
        })
        
        return {
            "content": text,
            "metadata": chunk_metadata,
            "id": f"{metadata.get('file_hash', 'unknown')}_{chunk_id}"
        }

# Global instances
_document_processor = None
_text_chunker = None

def get_document_processor() -> DocumentProcessor:
    """Get the global document processor instance"""
    global _document_processor
    if _document_processor is None:
        _document_processor = DocumentProcessor()
    return _document_processor

def get_text_chunker() -> TextChunker:
    """Get the global text chunker instance"""
    global _text_chunker
    if _text_chunker is None:
        _text_chunker = TextChunker()
    return _text_chunker
