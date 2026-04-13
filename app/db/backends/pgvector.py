"""
GSRS MCP Server - pgvector Backend Implementation
"""
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID

from sqlalchemy import create_engine, text, distinct, func, update
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import insert, JSONB

from app.db.base import VectorDatabase
from app.models import Base, VectorDocument, DBQueryResult


class PGVectorDatabase(VectorDatabase):
    """
    pgvector implementation of the VectorDatabase interface.

    Uses PostgreSQL with pgvector extension for vector storage and similarity search.
    """

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.SessionLocal = None

    def connect(self) -> None:
        """Establish connection to PostgreSQL."""
        self.engine = create_engine(self.database_url, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

    def disconnect(self) -> None:
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            self.engine = None
            self.SessionLocal = None

    def initialize(self, dimension: int = 384) -> None:
        """Create tables and indexes."""
        if self.engine is None:
            self.connect()

        if self.engine is None:
            raise ConnectionError("Failed to connect to the database.")

        with self.engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()

        Base.metadata.create_all(bind=self.engine)

    def _build_search_text(self, doc: VectorDocument) -> str:
        """Build denormalized search text for a document."""
        parts = [doc.text]
        metadata = doc.metadata_json or {}

        for key in ["canonical_name", "chunk_type", "section", "source_url"]:
            val = metadata.get(key)
            if val:
                parts.append(str(val))

        # Add names
        names = metadata.get("names", [])
        if isinstance(names, list):
            parts.extend(str(n) for n in names)

        # Add codes
        codes = metadata.get("codes", [])
        if isinstance(codes, list):
            for code in codes:
                if isinstance(code, dict):
                    code_text = code.get("code", "")
                    if code_text:
                        parts.append(str(code_text))
                elif isinstance(code, str):
                    parts.append(code)

        return " ".join(str(p) for p in parts if p)

    def _get_session(self) -> Session:
        """Get a database session."""
        if self.SessionLocal is None:
            self.connect()
        if self.SessionLocal is None:
            raise ConnectionError("Database session is not available.")
        return self.SessionLocal()

    def upsert_documents(self, documents: List[VectorDocument]) -> int:
        """Insert or update documents."""
        session = self._get_session()
        count = 0

        try:
            for doc in documents:
                # Build search_text if not set
                if not doc.search_text:
                    doc.search_text = self._build_search_text(doc)

                stmt = insert(VectorDocument).values(
                    chunk_id = doc.chunk_id,
                    **doc.values()
                ).on_conflict_do_update(
                    index_elements=[VectorDocument.chunk_id],
                    set_ = doc.values()
                )

                session.execute(stmt)

                count += 1

            session.commit()
            return count

        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[DBQueryResult]:
        """Search for similar documents using cosine similarity."""
        session = self._get_session()

        try:
            query = session.query(
                VectorDocument,
                VectorDocument.embedding.cosine_distance(query_embedding).label('similarity')
            )

            if filters:
                if 'section' in filters:
                    query = query.filter(
                        VectorDocument.section == filters['section']
                    )
                if 'document_id' in filters:
                    query = query.filter(
                        VectorDocument.document_id == filters['document_id']
                    )

            results = (
                query
                .order_by('similarity')
                .limit(top_k)
                .all()
            )

            query_results = []
            for chunk, similarity in results:
                query_results.append(DBQueryResult(document=chunk, score=1 - similarity))

            return query_results

        finally:
            session.close()

    def lexical_search(
        self,
        query: str,
        top_k: int = 40,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[DBQueryResult]:
        """
        Search for documents using PostgreSQL full-text search.
        Uses tsvector/tsquery with ranking.
        """
        session = self._get_session()

        try:
            # Build a tsquery from the search terms
            terms = query.lower().split()
            terms = [t for t in terms if len(t) >= 2]
            if not terms:
                return []

            search_expr = " & ".join(terms)

            # Build the plainto_tsquery expression
            tsquery = func.plainto_tsquery("english", search_expr)

            # Build the query using full-text search
            rank_col = func.ts_rank_cd(VectorDocument.search_tsv, tsquery).label("rank")

            q = session.query(
                VectorDocument,
                rank_col,
            ).filter(
                VectorDocument.search_tsv.op("@@")(tsquery),
            )

            # Apply filters
            if filters:
                if "section" in filters:
                    q = q.filter(VectorDocument.section == filters["section"])
                if "document_id" in filters:
                    q = q.filter(VectorDocument.document_id == filters["document_id"])

            results = q.order_by(rank_col.desc()).limit(top_k).all()

            # Build result list
            query_results = []
            for doc, rank in results:
                score = min(float(rank) if rank else 0.0, 1.0)
                query_results.append(DBQueryResult(document=doc, score=score))

            return query_results

        finally:
            session.close()

    def search_by_example(
        self,
        example: Dict[str, Any],
        top_k: int = 20,
        mode: str = "match",
    ) -> List[DBQueryResult]:
        """
        Search for documents whose metadata matches the given example JSON.

        Args:
            example: Example metadata to match against (e.g., {"canonical_name": "Aspirin"})
            top_k: Maximum number of results
            mode: Matching strategy:
                - "match": all top-level keys in example must match with same values
                - "contains": at least one key must match (broader search)
                - "nested": supports nested object matching with JSONB @> operator

        Returns:
            List of DBQueryResult ordered by match specificity (more matched keys = higher score)
        """
        if not example:
            return []

        session = self._get_session()

        try:
            import json
            from sqlalchemy import or_

            example_json = json.dumps(example)

            # Use PostgreSQL JSONB containment operator @>
            metadata_col = VectorDocument.metadata_json.cast(JSONB)

            if mode in ("match", "nested"):
                # Require all example keys to be present in metadata
                containment_filter = metadata_col.op("@>")(example_json)
                q = session.query(VectorDocument).filter(containment_filter)
            else:
                # "contains" mode — match if ANY key matches
                conditions = []
                for key, value in example.items():
                    if isinstance(value, (dict, list)):
                        nested_json = json.dumps({key: value})
                        conditions.append(metadata_col.op("@>")(nested_json))
                    else:
                        conditions.append(
                            metadata_col[key].as_string() == str(value)
                        )

                q = session.query(VectorDocument).filter(or_(*conditions))

            results = q.limit(top_k).all()

            # Score documents in Python
            query_results = []
            for doc in results:
                score = self._compute_example_match_score(
                    doc.metadata_json or {}, example, mode
                )
                query_results.append(DBQueryResult(document=doc, score=score))

            # Sort by score descending
            query_results.sort(key=lambda r: r.score, reverse=True)
            return query_results

        finally:
            session.close()

    def _compute_example_match_score(
        self,
        doc_metadata: Dict[str, Any],
        example: Dict[str, Any],
        mode: str,
    ) -> float:
        """
        Compute a match score between document metadata and example.
        Uses priority-based weighting (higher priority fields = higher score impact).

        Priority order (most to least reliable):
        1. UUID
        2. approvalID
        3. Reliable codes (UNII, SMS_ID, xEVMPD, ASK, ASKP)
        4. Definitional info (structure, InChI, SMILES, sequences)
        5. Systematic names
        6. Official names

        Returns a score between 0.0 and 1.0.
        """
        if not example:
            return 0.0

        # Priority weights (must sum to 1.0)
        priority_weights = {
            "uuid": 0.35,           # Priority 1: Highest reliability
            "approvalID": 0.20,     # Priority 2: Official approval ID
            "reliable_codes": 0.20, # Priority 3: Reliable identifier codes
            "structure": 0.10,      # Priority 4: Definitional information
            "systematic_names": 0.08,  # Priority 5: Systematic names
            "official_names": 0.05,    # Priority 6: Official names
            "all_codes": 0.02,         # Lower priority: other codes
            "other_names": 0.02,       # Lower priority: other names
            "canonical_name": 0.05,    # Fallback name matching
            "classifications": 0.03,   # Additional context
        }

        total_score = 0.0
        total_weight = 0.0

        for key, example_value in example.items():
            weight = priority_weights.get(key, 0.02)  # Default low weight for unknown fields
            total_weight += weight

            if key not in doc_metadata:
                continue

            doc_value = doc_metadata[key]
            key_score = 0.0

            if key == "uuid":
                # Exact match only
                key_score = 1.0 if str(doc_value) == str(example_value) else 0.0

            elif key == "approvalID":
                # Exact match
                key_score = 1.0 if str(doc_value) == str(example_value) else 0.0

            elif key == "reliable_codes":
                # Match any reliable code
                if isinstance(example_value, dict) and isinstance(doc_value, dict):
                    matched = 0
                    total = len(example_value)
                    for code_system, code_val in example_value.items():
                        if code_system in doc_value and str(doc_value[code_system]) == str(code_val):
                            matched += 1
                    key_score = matched / total if total > 0 else 0.0

            elif key == "structure":
                # Partial match on structure fields
                if isinstance(example_value, dict) and isinstance(doc_value, dict):
                    matched = 0
                    total = len(example_value)
                    for field, val in example_value.items():
                        if field in doc_value and str(doc_value[field]) == str(val):
                            matched += 1
                    key_score = matched / total if total > 0 else 0.0

            elif key in ("systematic_names", "official_names", "other_names"):
                # Name matching - exact or partial
                if isinstance(example_value, list) and isinstance(doc_value, list):
                    example_set = set(str(n).lower() for n in example_value)
                    doc_set = set(str(n).lower() for n in doc_value)
                    if example_set & doc_set:
                        key_score = 1.0
                    elif any(ex in str(doc_value).lower() for ex in example_set):
                        key_score = 0.7

            elif key == "canonical_name":
                # Fallback name matching
                if str(doc_value).lower() == str(example_value).lower():
                    key_score = 1.0
                elif str(example_value).lower() in str(doc_value).lower():
                    key_score = 0.7

            elif key == "all_codes":
                # Code matching
                if isinstance(example_value, dict) and isinstance(doc_value, dict):
                    matched = 0
                    total = len(example_value)
                    for code_system, code_val in example_value.items():
                        if code_system in doc_value and str(doc_value[code_system]) == str(code_val):
                            matched += 1
                    key_score = matched / total if total > 0 else 0.0

            elif key == "classifications":
                # Classification matching
                if isinstance(example_value, list) and isinstance(doc_value, list):
                    example_set = set(str(c).lower() for c in example_value)
                    doc_set = set(str(c).lower() for c in doc_value)
                    if example_set & doc_set:
                        key_score = 1.0

            else:
                # Generic matching for unknown fields
                if isinstance(example_value, dict) and isinstance(doc_value, dict):
                    if self._deep_match(doc_value, example_value):
                        key_score = 1.0
                elif isinstance(example_value, list) and isinstance(doc_value, list):
                    if set(str(v) for v in example_value) & set(str(v) for v in doc_value):
                        key_score = 1.0
                else:
                    if str(doc_value) == str(example_value):
                        key_score = 1.0

            total_score += weight * key_score

        # Normalize by total weight
        if mode == "contains":
            return 1.0 if total_score > 0 else 0.0

        return total_score / total_weight if total_weight > 0 else 0.0

    def _deep_match(self, doc_obj: Dict, example_obj: Dict) -> bool:
        """Check if doc_obj contains all keys/values from example_obj."""
        for key, val in example_obj.items():
            if key not in doc_obj:
                return False
            if isinstance(val, dict):
                if not isinstance(doc_obj[key], dict):
                    return False
                if not self._deep_match(doc_obj[key], val):
                    return False
            elif str(doc_obj[key]) != str(val):
                return False
        return True

    def get_document(self, doc_id: str) -> Optional[VectorDocument]:
        """Get a document by chunk_id."""
        session = self._get_session()

        try:
            chunk = session.query(VectorDocument).filter(
                VectorDocument.chunk_id == doc_id
            ).first()

            if chunk:
                return chunk
            return None
        finally:
            session.close()

    def get_documents_by_substance(
        self,
        substance_uuid: UUID,
        limit: Optional[int] = None
    ) -> List[VectorDocument]:
        """Get all documents for a substance."""
        session = self._get_session()
        try:
            query = session.query(VectorDocument).filter(
                VectorDocument.document_id == substance_uuid
            )

            if limit:
                query = query.limit(limit)

            chunks = query.all()

            return chunks
        finally:
            session.close()

    def delete_documents_by_substance(self, substance_uuid: UUID) -> int:
        """Delete all documents for a substance."""
        session = self._get_session()
        try:
            count = session.query(VectorDocument).filter(
                VectorDocument.document_id == substance_uuid
            ).delete(synchronize_session=False)
            session.commit()
            return count
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def delete_all(self) -> None:
        """Delete all documents."""
        session = self._get_session()
        try:
            session.query(VectorDocument).delete()
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_statistics(self) -> Dict[str, int]:
        """Get database statistics."""
        session = self._get_session()
        try:
            total_chunks = session.query(VectorDocument).count()
            substances = session.query(
                VectorDocument.document_id
            ).distinct().count()

            return {
                "total_chunks": total_chunks,
                "total_substances": substances,
            }
        finally:
            session.close()

    def get_unique_values(self, field: str) -> List[str]:
        """Get unique values for a field."""
        session = self._get_session()
        try:
            if field == "section":
                results = session.query(distinct(VectorDocument.section)).all()
                return [r[0] for r in results if r[0]]
            elif field == "source_url":
                results = session.query(distinct(VectorDocument.source_url)).all()
                return [r[0] for r in results if r[0]]
            return []
        finally:
            session.close()

