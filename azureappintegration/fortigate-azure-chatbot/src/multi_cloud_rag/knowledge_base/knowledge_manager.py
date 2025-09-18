"""
Knowledge Base Manager
Manages multi-cloud VM architecture knowledge base with metadata tagging
"""

import os
import json
import yaml
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import hashlib
from pathlib import Path

from ..vector_stores import DocumentMetadata, CloudProvider, DocumentType

logger = logging.getLogger(__name__)

@dataclass
class KnowledgeDocument:
    """Knowledge base document structure"""
    id: str
    title: str
    content: str
    metadata: DocumentMetadata
    tags: List[str]
    created_at: str
    updated_at: str
    source_file: Optional[str] = None
    checksum: Optional[str] = None

class KnowledgeBaseManager:
    """Manages multi-cloud knowledge base documents"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.documents_path = self.base_path / "documents"
        self.metadata_path = self.base_path / "metadata"
        self.index_file = self.base_path / "index.json"
        
        # Create directory structure
        self._create_directory_structure()
        
        # Load existing index
        self.document_index = self._load_index()
    
    def _create_directory_structure(self):
        """Create knowledge base directory structure"""
        directories = [
            self.documents_path / "azure" / "vm-config",
            self.documents_path / "azure" / "networking",
            self.documents_path / "azure" / "security",
            self.documents_path / "azure" / "best-practices",
            self.documents_path / "gcp" / "vm-config",
            self.documents_path / "gcp" / "networking",
            self.documents_path / "gcp" / "security",
            self.documents_path / "gcp" / "best-practices",
            self.documents_path / "multi-cloud" / "architecture",
            self.documents_path / "multi-cloud" / "cost-optimization",
            self.documents_path / "multi-cloud" / "identity",
            self.metadata_path
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_index(self) -> Dict[str, Any]:
        """Load document index"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load index: {e}")
        
        return {"documents": {}, "last_updated": datetime.now().isoformat()}
    
    def _save_index(self):
        """Save document index"""
        try:
            self.document_index["last_updated"] = datetime.now().isoformat()
            with open(self.index_file, 'w') as f:
                json.dump(self.document_index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def _generate_document_id(self, title: str, cloud: str, topic: str) -> str:
        """Generate unique document ID"""
        content = f"{cloud}-{topic}-{title}".lower().replace(" ", "-")
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _calculate_checksum(self, content: str) -> str:
        """Calculate content checksum"""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def add_document(
        self,
        title: str,
        content: str,
        metadata: DocumentMetadata,
        tags: Optional[List[str]] = None,
        source_file: Optional[str] = None
    ) -> str:
        """Add document to knowledge base"""
        try:
            # Generate document ID
            doc_id = self._generate_document_id(title, metadata.cloud.value, metadata.topic.value)
            
            # Create document
            document = KnowledgeDocument(
                id=doc_id,
                title=title,
                content=content,
                metadata=metadata,
                tags=tags or [],
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                source_file=source_file,
                checksum=self._calculate_checksum(content)
            )
            
            # Save document file
            doc_path = self._get_document_path(metadata.cloud.value, metadata.topic.value)
            doc_file = doc_path / f"{doc_id}.json"
            
            with open(doc_file, 'w') as f:
                json.dump(asdict(document), f, indent=2, default=str)
            
            # Update index
            self.document_index["documents"][doc_id] = {
                "title": title,
                "cloud": metadata.cloud.value,
                "topic": metadata.topic.value,
                "file_path": str(doc_file),
                "tags": tags or [],
                "created_at": document.created_at,
                "updated_at": document.updated_at
            }
            
            self._save_index()
            logger.info(f"Added document: {title} ({doc_id})")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            raise
    
    def get_document(self, doc_id: str) -> Optional[KnowledgeDocument]:
        """Get document by ID"""
        try:
            if doc_id not in self.document_index["documents"]:
                return None
            
            doc_info = self.document_index["documents"][doc_id]
            doc_file = Path(doc_info["file_path"])
            
            if not doc_file.exists():
                logger.warning(f"Document file not found: {doc_file}")
                return None
            
            with open(doc_file, 'r') as f:
                doc_data = json.load(f)
            
            # Reconstruct metadata
            metadata_dict = doc_data["metadata"]
            metadata = DocumentMetadata(
                cloud=CloudProvider(metadata_dict["cloud"]),
                topic=DocumentType(metadata_dict["topic"]),
                region=metadata_dict["region"],
                complexity=metadata_dict["complexity"],
                use_case=metadata_dict["use_case"],
                last_updated=metadata_dict["last_updated"],
                source_url=metadata_dict.get("source_url"),
                version=metadata_dict.get("version"),
                compliance=metadata_dict.get("compliance")
            )
            
            return KnowledgeDocument(
                id=doc_data["id"],
                title=doc_data["title"],
                content=doc_data["content"],
                metadata=metadata,
                tags=doc_data["tags"],
                created_at=doc_data["created_at"],
                updated_at=doc_data["updated_at"],
                source_file=doc_data.get("source_file"),
                checksum=doc_data.get("checksum")
            )
            
        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            return None
    
    def update_document(self, doc_id: str, **kwargs) -> bool:
        """Update document"""
        try:
            document = self.get_document(doc_id)
            if not document:
                logger.error(f"Document not found: {doc_id}")
                return False
            
            # Update fields
            for key, value in kwargs.items():
                if hasattr(document, key):
                    setattr(document, key, value)
            
            # Update timestamp and checksum
            document.updated_at = datetime.now().isoformat()
            if 'content' in kwargs:
                document.checksum = self._calculate_checksum(document.content)
            
            # Save updated document
            doc_info = self.document_index["documents"][doc_id]
            doc_file = Path(doc_info["file_path"])
            
            with open(doc_file, 'w') as f:
                json.dump(asdict(document), f, indent=2, default=str)
            
            # Update index
            self.document_index["documents"][doc_id]["updated_at"] = document.updated_at
            self._save_index()
            
            logger.info(f"Updated document: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document {doc_id}: {e}")
            return False
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document"""
        try:
            if doc_id not in self.document_index["documents"]:
                logger.warning(f"Document not found in index: {doc_id}")
                return False
            
            doc_info = self.document_index["documents"][doc_id]
            doc_file = Path(doc_info["file_path"])
            
            # Delete file
            if doc_file.exists():
                doc_file.unlink()
            
            # Remove from index
            del self.document_index["documents"][doc_id]
            self._save_index()
            
            logger.info(f"Deleted document: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False
    
    def search_documents(
        self,
        query: Optional[str] = None,
        cloud: Optional[CloudProvider] = None,
        topic: Optional[DocumentType] = None,
        tags: Optional[List[str]] = None,
        complexity: Optional[str] = None
    ) -> List[str]:
        """Search documents by criteria"""
        matching_docs = []
        
        for doc_id, doc_info in self.document_index["documents"].items():
            # Filter by cloud
            if cloud and doc_info["cloud"] != cloud.value:
                continue
            
            # Filter by topic
            if topic and doc_info["topic"] != topic.value:
                continue
            
            # Filter by tags
            if tags:
                doc_tags = set(doc_info.get("tags", []))
                if not any(tag in doc_tags for tag in tags):
                    continue
            
            # Text search in title
            if query:
                if query.lower() not in doc_info["title"].lower():
                    continue
            
            matching_docs.append(doc_id)
        
        return matching_docs
    
    def get_all_documents(self) -> List[KnowledgeDocument]:
        """Get all documents"""
        documents = []
        for doc_id in self.document_index["documents"]:
            doc = self.get_document(doc_id)
            if doc:
                documents.append(doc)
        return documents
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        total_docs = len(self.document_index["documents"])
        
        cloud_counts = {}
        topic_counts = {}
        
        for doc_info in self.document_index["documents"].values():
            cloud = doc_info["cloud"]
            topic = doc_info["topic"]
            
            cloud_counts[cloud] = cloud_counts.get(cloud, 0) + 1
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        return {
            "total_documents": total_docs,
            "by_cloud": cloud_counts,
            "by_topic": topic_counts,
            "last_updated": self.document_index["last_updated"]
        }
    
    def _get_document_path(self, cloud: str, topic: str) -> Path:
        """Get path for document storage"""
        return self.documents_path / cloud / topic
    
    def import_from_directory(self, source_dir: str, cloud: CloudProvider, topic: DocumentType):
        """Import documents from directory"""
        source_path = Path(source_dir)
        if not source_path.exists():
            logger.error(f"Source directory not found: {source_dir}")
            return
        
        imported_count = 0
        
        for file_path in source_path.rglob("*.md"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract title from filename
                title = file_path.stem.replace('-', ' ').replace('_', ' ').title()
                
                # Create metadata
                metadata = DocumentMetadata(
                    cloud=cloud,
                    topic=topic,
                    region="global",
                    complexity="intermediate",
                    use_case="general",
                    last_updated=datetime.now().isoformat()
                )
                
                # Add document
                self.add_document(
                    title=title,
                    content=content,
                    metadata=metadata,
                    source_file=str(file_path)
                )
                
                imported_count += 1
                
            except Exception as e:
                logger.error(f"Failed to import {file_path}: {e}")
        
        logger.info(f"Imported {imported_count} documents from {source_dir}")
    
    def export_to_format(self, output_dir: str, format: str = "json"):
        """Export knowledge base to specified format"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        documents = self.get_all_documents()
        
        if format == "json":
            export_data = [asdict(doc) for doc in documents]
            with open(output_path / "knowledge_base.json", 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        elif format == "yaml":
            export_data = [asdict(doc) for doc in documents]
            with open(output_path / "knowledge_base.yaml", 'w') as f:
                yaml.dump(export_data, f, default_flow_style=False)
        
        elif format == "markdown":
            for doc in documents:
                filename = f"{doc.metadata.cloud.value}_{doc.metadata.topic.value}_{doc.id}.md"
                filepath = output_path / filename
                
                with open(filepath, 'w') as f:
                    f.write(f"# {doc.title}\n\n")
                    f.write(f"**Cloud:** {doc.metadata.cloud.value}\n")
                    f.write(f"**Topic:** {doc.metadata.topic.value}\n")
                    f.write(f"**Region:** {doc.metadata.region}\n")
                    f.write(f"**Complexity:** {doc.metadata.complexity}\n")
                    f.write(f"**Use Case:** {doc.metadata.use_case}\n")
                    f.write(f"**Tags:** {', '.join(doc.tags)}\n\n")
                    f.write(doc.content)
        
        logger.info(f"Exported knowledge base to {output_dir} in {format} format")
