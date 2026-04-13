"""
GSRS MCP Server - ChromaDB Backend Implementation

ChromaDB is a lightweight, embedded vector database perfect for
development and testing. It requires no external server.
"""
import os
from typing import List, Dict, Any, Optional, TYPE_CHECKING, Sequence
from uuid import UUID
import json

if TYPE_CHECKING:
    import chromadb
    from chromadb.config import Settings
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection
    from chromadb.base_types import Vector
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

from app.db.base import VectorDatabase
from app.models import VectorDocument, DBQueryResult

class ChromaDatabase(VectorDatabase):
    """
    ChromaDB implementation of the VectorDatabase interface.

    Uses ChromaDB for local, serverless vector storage.
    Ideal for development and testing.
    """

    def __init__(self, database_url: str = "chroma://./chroma_data/chunks"):
        if not CHROMA_AVAILABLE:
            raise ImportError(
                "ChromaDB is not installed. Install with: pip install chromadb"
            )

        # Parse chroma URL: chroma://<persist_directory>/<collection_name>
        from urllib.parse import urlparse
        parsed = urlparse(database_url)

        # Reconstruct path with netloc (for cases like chroma://./path/collection)
        path = parsed.netloc + parsed.path
        path = os.path.normpath(path.lstrip("/"))

        # Split path into directory and collection name
        persist_directory, collection_name = os.path.split(path)
        if persist_directory and collection_name:
            self.persist_directory = persist_directory
            self.collection_name = collection_name
        else:
            self.persist_directory = path or "./chroma_data"
            self.collection_name = "chunks"

        self.client: Optional["ClientAPI"] = None
        self.collection: Optional["Collection"] = None
    
    def connect(self) -> None:
        """Initialize ChromaDB client."""
        # Use persistent storage
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
    
    def disconnect(self) -> None:
        """Close ChromaDB connection."""
        client = self.client
        self.collection = None
        self.client = None

        if client is None:
            return

        close_method = getattr(client, "close", None)
        if callable(close_method):
            try:
                close_method()
            except Exception:
                pass

        clear_cache = getattr(client, "clear_system_cache", None)
        if callable(clear_cache):
            try:
                clear_cache()
            except Exception:
                pass
    
    def initialize(self, dimension: int = 384) -> None:
        """Create or get the collection."""
        if self.client is None:
            self.connect()
        if self.client is None:
            raise RuntimeError("Failed to connect to ChromaDB.")

        # Preserve existing data across restarts and validate dimensions when possible.
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"dimension": dimension}
        )
        existing_dimension = (self.collection.metadata or {}).get("dimension")
        if existing_dimension and int(existing_dimension) != int(dimension):
            raise RuntimeError(
                "ChromaDB collection dimension mismatch: "
                f"existing={existing_dimension}, configured={dimension}. "
                "Use a fresh collection path or align EMBEDDING_DIMENSION."
            )
    
    def upsert_documents(self, documents: List[VectorDocument]) -> int:
        """Insert or update documents."""
        if self.collection is None:
            raise RuntimeError("Database not connected. Call connect() first.")

        ids = []
        embeddings = []
        metadatas = []
        documents_list = []

        for doc in documents:
            # Use chunk_id as the ChromaDB ID for consistent retrieval
            ids.append(doc.chunk_id)
            embeddings.append(doc.embedding)
            metadatas.append({
                "document_id": str(doc.document_id),
                "chunk_id": doc.chunk_id,
                "section": doc.section,
                "source_url": doc.source_url or "",
                "metadata_json": json.dumps(doc.metadata_json)
            })
            documents_list.append(doc.text)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents_list
        )

        return len(documents)
    
    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[DBQueryResult]:
        """Search for similar documents."""
        if self.collection is None:
            raise RuntimeError("Database not connected. Call connect() first.")

        # Build where filter
        where = None
        if filters:
            where = {}
            if "section" in filters:
                where["section"] = filters["section"]
            if "document_id" in filters:
                where["document_id"] = filters["document_id"]
            
            # ChromaDB requires non-empty where dict
            if not where:
                where = None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["embeddings", "metadatas", "documents", "distances"]
        )

        query_results = []

        if results and results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}

                # Parse metadata JSON
                metadata_json = {}
                if 'metadata_json' in metadata:
                    metadata_value = metadata['metadata_json']
                    if isinstance(metadata_value, str):
                        try:
                            metadata_json = json.loads(metadata_value)
                        except (json.JSONDecodeError, TypeError):
                            metadata_json = {}

                # Handle embeddings - ChromaDB may return numpy arrays
                embedding: List[float] = []
                if results['embeddings'] and i < len(results['embeddings'][0]):
                    emb = results['embeddings'][0][i]
                    embedding = [] if emb is None else emb.tolist() if not isinstance(emb, Sequence) else list(emb)

                chunk = VectorDocument(
                    chunk_id=str(metadata.get('chunk_id', doc_id)),
                    document_id=UUID(str(metadata.get('document_id', '00000000-0000-0000-0000-000000000000'))),
                    section=metadata.get('section', ''),
                    source_url=metadata.get('source_url', ''),
                    text=results['documents'][0][i] if results['documents'] else '',
                    embedding=embedding,
                    metadata_json=metadata_json
                )

                # Chroma returns distance, convert to similarity score
                distance = results['distances'][0][i] if results['distances'] else 0
                score = 1 - distance  # Convert distance to similarity

                query_results.append(DBQueryResult(document=chunk, score=score))

        return query_results
    
    def get_document(self, doc_id: str) -> Optional[VectorDocument]:
        """Get a document by ID."""
        if self.collection is None:
            raise RuntimeError("Database not connected. Call connect() first.")

        results = self.collection.get(
            ids=[doc_id],
            include=["embeddings", "metadatas", "documents"]
        )

        # ChromaDB returns flat lists: {'ids': ['id1'], 'metadatas': [meta1], ...}
        if not results or not results['ids'] or len(results['ids']) == 0:
            return None

        metadata = results['metadatas'][0] if results['metadatas'] and len(results['metadatas']) > 0 else {}

        metadata_json = {}
        if 'metadata_json' in metadata:
            metadata_value = metadata['metadata_json']
            if isinstance(metadata_value, str):
                try:
                    metadata_json = json.loads(metadata_value)
                except (json.JSONDecodeError, TypeError):
                    metadata_json = {}

        # Handle embeddings - ChromaDB may return numpy arrays
        embedding: List[float] = []
        if results['embeddings'] is not None and len(results['embeddings']) > 0:
            emb = results['embeddings'][0]
            embedding = [] if emb is None else emb.tolist() if not isinstance(emb, Sequence) else list(emb)

        return VectorDocument(
            chunk_id=metadata.get('chunk_id', results['ids'][0]),
            document_id=UUID(str(metadata.get('document_id', '00000000-0000-0000-0000-000000000000'))),
            section=metadata.get('section', ''),
            source_url=metadata.get('source_url', ''),
            text=results['documents'][0] if results['documents'] and len(results['documents']) > 0 else '',
            embedding=embedding,
            metadata_json=metadata_json
        )

    def get_documents_by_substance(
        self,
        substance_uuid: UUID,
        limit: Optional[int] = None
    ) -> List[VectorDocument]:
        """Get all documents for a substance."""
        if self.collection is None:
            raise RuntimeError("Database not connected. Call connect() first.")

        results = self.collection.get(
            where={"document_id": str(substance_uuid)},
            include=["embeddings", "metadatas", "documents"]
        )

        documents = []
        count = 0

        if results and results['ids']:
            for i, doc_id in enumerate(results['ids']):
                if limit and count >= limit:
                    break

                metadata = results['metadatas'][i] if results['metadatas'] and i < len(results['metadatas']) else {}

                metadata_json = {}
                if 'metadata_json' in metadata:
                    metadata_value = metadata['metadata_json']
                    if isinstance(metadata_value, str):
                        try:
                            metadata_json = json.loads(metadata_value)
                        except (json.JSONDecodeError, TypeError):
                            metadata_json = {}

                # Handle embeddings - ChromaDB may return numpy arrays
                embedding: List[float] = []
                if results['embeddings'] is not None and i < len(results['embeddings']):
                    emb = results['embeddings'][i]
                    embedding = [] if emb is None else emb.tolist() if not isinstance(emb, Sequence) else list(emb)

                documents.append(VectorDocument(
                    chunk_id=metadata.get('chunk_id', doc_id),
                    document_id=UUID(str(metadata.get('document_id', '00000000-0000-0000-0000-000000000000'))),
                    section=metadata.get('section', ''),
                    source_url=metadata.get('source_url', ''),
                    text=results['documents'][i] if results['documents'] and i < len(results['documents']) else '',
                    embedding=embedding,
                    metadata_json=metadata_json
                ))
                count += 1

        return documents
    
    def delete_documents_by_substance(self, substance_uuid: UUID) -> int:
        """Delete all documents for a substance."""
        if self.collection is None:
            raise RuntimeError("Database not connected. Call connect() first.")

        # First get count
        results = self.collection.get(
            where={"document_id": str(substance_uuid)},
            include=[]
        )

        count = len(results['ids']) if results and results['ids'] else 0

        # Delete
        self.collection.delete(
            where={"document_id": str(substance_uuid)}
        )

        return count
    
    def delete_all(self) -> None:
        """Delete all documents."""
        if self.collection is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        # Delete and recreate collection
        if self.client is not None:
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.create_collection(
                name=self.collection.name,
                metadata=self.collection.metadata
            )
    
    def get_statistics(self) -> Dict[str, int]:
        """Get database statistics."""
        if self.collection is None:
            raise RuntimeError("Database not connected. Call connect() first.")

        # Get all metadata to calculate statistics
        results = self.collection.get(
            include=["metadatas"]
        )

        total_chunks = len(results['ids']) if results and results['ids'] else 0

        document_ids = set()

        if results and results['metadatas']:
            for metadata in results['metadatas']:
                if 'document_id' in metadata:
                    document_ids.add(metadata['document_id'])

        return {
            "total_chunks": total_chunks,
            "total_substances": len(document_ids),
        }
    
    def get_unique_values(self, field: str) -> List[str]:
        """Get unique values for a field."""
        if self.collection is None:
            raise RuntimeError("Database not connected. Call connect() first.")

        results = self.collection.get(include=["metadatas"])

        values = set()
        if results and results['metadatas']:
            for metadata in results['metadatas']:
                # Check both direct metadata and parsed metadata_json
                if field in metadata and metadata[field]:
                    values.add(metadata[field])
                elif 'metadata_json' in metadata:
                    # Parse metadata_json if present
                    metadata_json = {}
                    metadata_value = metadata['metadata_json']
                    if isinstance(metadata_value, str):
                        try:
                            metadata_json = json.loads(metadata_value)
                        except (json.JSONDecodeError, TypeError):
                            metadata_json = {}
                    elif isinstance(metadata_value, dict):
                        metadata_json = metadata_value
                    value =  metadata_json.get(field)
                    if value:
                        values.add(value)
        return sorted(list(values))

    def lexical_search(
        self,
        query: str,
        top_k: int = 40,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[DBQueryResult]:
        """
        Lexical search fallback for ChromaDB.
        Uses simple token matching and substring scoring.
        """
        if self.collection is None:
            raise RuntimeError("Database not connected. Call connect() first.")

        import re

        terms = re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", query.lower())
        terms = [t for t in terms if len(t) >= 2]

        if not terms:
            return []

        # Get all documents (with optional metadata filter)
        where = None
        if filters:
            where = {}
            if "section" in filters:
                where["section"] = filters["section"]
            if "document_id" in filters:
                where["document_id"] = str(filters["document_id"])
            if not where:
                where = None

        # Fetch all documents
        results = self.collection.get(
            where=where,
            include=["metadatas", "documents"]
        )

        if not results or not results['ids']:
            return []

        scored = []
        for i, doc_id in enumerate(results['ids']):
            text = results['documents'][i] if results['documents'] else ""
            metadata = results['metadatas'][i] if results['metadatas'] else {}

            # Parse metadata JSON
            metadata_json = {}
            if 'metadata_json' in metadata:
                try:
                    metadata_json = json.loads(metadata['metadata_json'])
                except (json.JSONDecodeError, TypeError):
                    pass

            # Reconstruct VectorDocument
            embedding: List[float] = []
            doc = VectorDocument(
                chunk_id=metadata.get('chunk_id', doc_id),
                document_id=UUID(str(metadata.get('document_id', '00000000-0000-0000-0000-000000000000'))),
                section=metadata.get('section', ''),
                source_url=metadata.get('source_url', ''),
                text=text,
                embedding=embedding,
                metadata_json=metadata_json,
            )

            # Score based on term overlap
            score = self._score_lexical_match(text, metadata, metadata_json, terms)
            if score > 0:
                scored.append(DBQueryResult(document=doc, score=score))

        # Sort by score descending
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def _score_lexical_match(self, text: str, metadata: Dict, metadata_json: Dict, terms: List[str]) -> float:
        """Score a document based on lexical term overlap."""
        text_lower = text.lower()

        # Build search text from text + metadata
        search_parts = [text_lower]
        for key in ["canonical_name", "chunk_type", "section"]:
            val = metadata.get(key) or metadata_json.get(key)
            if val:
                search_parts.append(str(val).lower())

        search_text = " ".join(search_parts)

        matched = 0
        for term in terms:
            if term in search_text:
                matched += 1

        if not terms:
            return 0.0

        # Normalize
        score = matched / len(terms)

        # Boost for exact phrase match
        phrase = " ".join(terms)
        if phrase in search_text:
            score += 0.5

        return min(score, 1.0)

    def search_by_example(
        self,
        example: Dict[str, Any],
        top_k: int = 20,
        mode: str = "match",
    ) -> List[DBQueryResult]:
        """
        Search for documents matching example metadata in ChromaDB.

        Uses ChromaDB's where-clause filtering for exact matches and
        scores documents by how many example keys matched.

        Args:
            example: Metadata dict to match (e.g. {"canonical_name": "Aspirin"})
            top_k: Maximum results
            mode: 'match' (all keys), 'contains' (any key), or 'nested'

        Returns:
            Ranked DBQueryResult list
        """
        if self.collection is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        if not example:
            return []

        # Fetch all documents (with optional pre-filter)
        results = self.collection.get(
            include=["metadatas", "documents", "embeddings"]
        )

        if not results or not results['ids']:
            return []

        scored = []
        for i, doc_id in enumerate(results['ids']):
            text = results['documents'][i] if results['documents'] else ""
            metadata_raw = results['metadatas'][i] if results['metadatas'] else {}

            # Parse metadata JSON
            metadata_json = {}
            if 'metadata_json' in metadata_raw:
                try:
                    metadata_json = json.loads(metadata_raw['metadata_json'])
                except (json.JSONDecodeError, TypeError):
                    pass

            # Compute match score
            score = self._compute_example_match_score(
                metadata_json, example, mode
            )

            if score > 0:
                embedding: List[float] = []
                emb_list = results.get('embeddings')
                if emb_list is not None and i < len(emb_list):
                    emb = emb_list[i]
                    embedding = (
                        [] if emb is None
                        else emb.tolist() if not isinstance(emb, Sequence)
                        else list(emb)
                    )

                doc = VectorDocument(
                    chunk_id=metadata_raw.get('chunk_id', doc_id),
                    document_id=UUID(str(
                        metadata_raw.get('document_id',
                                         '00000000-0000-0000-0000-000000000000')
                    )),
                    section=metadata_raw.get('section', ''),
                    source_url=metadata_raw.get('source_url', ''),
                    text=text,
                    embedding=embedding,
                    metadata_json=metadata_json,
                )
                scored.append(DBQueryResult(document=doc, score=score))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    @staticmethod
    def _compute_example_match_score(
        doc_metadata: Dict[str, Any],
        example: Dict[str, Any],
        mode: str,
    ) -> float:
        """
        Score document metadata against example with priority-based weights.

        Priority order (most to least reliable):
        1. UUID                     — 35%
        2. approvalID               — 20%
        3. reliable_codes           — 20%
        4. structure                — 10%
        5. systematic_names         —  8%
        6. official_names           —  5%
        7. canonical_name           —  5%
        8. classifications          —  3%
        9. all_codes, other_names   —  2% each
        """
        if not example or not doc_metadata:
            return 0.0

        weights = {
            "uuid": 0.35,
            "approvalID": 0.20,
            "reliable_codes": 0.20,
            "structure": 0.10,
            "systematic_names": 0.08,
            "official_names": 0.05,
            "canonical_name": 0.05,
            "classifications": 0.03,
            "all_codes": 0.02,
            "other_names": 0.02,
        }

        total_score = 0.0
        total_weight = 0.0

        for key, ex_val in example.items():
            w = weights.get(key, 0.02)
            total_weight += w

            if key not in doc_metadata:
                continue

            doc_val = doc_metadata[key]
            ks = 0.0

            if key in ("uuid", "approvalID", "canonical_name"):
                ks = 1.0 if str(doc_val) == str(ex_val) else 0.0

            elif key in ("reliable_codes", "all_codes", "structure"):
                if isinstance(ex_val, dict) and isinstance(doc_val, dict):
                    matched = sum(
                        1 for k, v in ex_val.items()
                        if k in doc_val and str(doc_val[k]) == str(v)
                    )
                    ks = matched / len(ex_val) if ex_val else 0.0

            elif key in ("systematic_names", "official_names",
                         "other_names", "classifications"):
                if isinstance(ex_val, list) and isinstance(doc_val, list):
                    ex_set = {str(x).lower() for x in ex_val}
                    doc_set = {str(x).lower() for x in doc_val}
                    if ex_set & doc_set:
                        ks = 1.0
                    elif any(e in str(doc_val).lower() for e in ex_set):
                        ks = 0.7

            else:
                # Generic fallback
                if isinstance(ex_val, dict) and isinstance(doc_val, dict):
                    if all(
                        str(doc_val.get(k)) == str(v)
                        for k, v in ex_val.items() if k in doc_val
                    ):
                        ks = 1.0
                elif isinstance(ex_val, list) and isinstance(doc_val, list):
                    if set(str(v) for v in ex_val) & set(str(v) for v in doc_val):
                        ks = 1.0
                elif str(doc_val) == str(ex_val):
                    ks = 1.0

            total_score += w * ks

        if mode == "contains":
            return 1.0 if total_score > 0 else 0.0

        return total_score / total_weight if total_weight > 0 else 0.0



