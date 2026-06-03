# Parent-Child Retrieval Implementation

## Overview

This implementation provides parent-child retrieval using **virtual parent reconstruction** from existing chunk records. This approach minimizes schema changes, migration effort, and reindexing while delivering most of the benefits of parent-child retrieval.

## Key Concepts

### Parent Identity
A parent is uniquely identified by:
- **document_id**: The UUID of the substance/document
- **root_section**: The topmost section within the document

Example: `ParentIdentity(document_id=uuid(...), root_section="root")`

### Virtual Parent Reconstruction
Instead of storing explicit parent-child relationships:
1. Parent context is **reconstructed on-demand** by loading all chunks sharing the same `(document_id, root_section)`
2. Chunks are enriched with this parent context
3. This provides broader document context without requiring schema modifications

## Architecture

### Core Components

#### 1. `ParentContextEnricher` (app/services/parent_child_retrieval.py)
Main service class that handles virtual parent reconstruction:

```python
enricher = ParentContextEnricher(vector_db)

# Extract root section from a chunk
root_section = enricher.extract_root_section(chunk)

# Get parent identity for a chunk
parent_id = enricher.get_parent_identity(chunk)

# Reconstruct parent context (loads all chunks in same parent)
parent_context = enricher.reconstruct_parent_context(parent_id)

# Enrich a chunk with parent context
enriched = enricher.enrich_chunk_with_parent(chunk)

# Enrich search results
enriched_results = enricher.enrich_search_results(search_results)
```

#### 2. Enhanced Vector Database Interface
New methods on `VectorDatabase` base class:

```python
# Get documents filtered by section and substance
docs = vector_db.get_documents_by_section_and_substance(
    substance_uuid=uuid,
    section="codes",
    limit=100
)
```

#### 3. Backend Implementations
- **PGVector**: Optimized SQL query for filtered retrieval
- **ChromaDB**: Filter-based query for filtered retrieval

#### 4. VectorDatabaseService Integration
The service layer exposes the parent enricher:

```python
# Access parent context enricher
enricher = vector_db.parent_enricher
enriched_results = enricher.enrich_search_results(results)
```

## MCP Tools

### 1. `rag_query_with_parent_context`
Enhanced RAG query that automatically enriches results with parent context.

**Parameters:**
- `query` (str): Search query
- `top_k` (int, default=8): Number of results
- `include_parent_text` (bool, default=True): Include parent text summary
- `parent_text_limit` (int, default=1000): Max characters for parent text
- `filters` (str): JSON filter object

**Returns:** Formatted results with parent context enrichment

**Example:**
```
Query: "aspirin mechanism"
Results include:
1. Score: 0.95 | Section: mechanisms
   Text: [child chunk text]
   Parent Context: 3 chunks in root, names, codes
   Parent Summary: [aggregated text from parent sections]
```

### 2. `get_parent_context`
Retrieve parent context for a specific chunk.

**Parameters:**
- `chunk_id` (str): The chunk ID (e.g., "root_uuid:12345678-...")

**Returns:** 
- Parent identity information
- Number of chunks in parent
- Sections included
- Text parts from parent chunks
- Unified metadata

**Example:**
```
Parent Context for Chunk: root_uuid:12345678-...
Document: <substance-uuid>
Root Section: root
Num Chunks in Parent: 5
Sections Included: root, names, codes, references
[Text from all parent chunks...]
```

### 3. Standard RAG Tools (Unchanged)
- `rag_query`: Traditional RAG query (still available)
- `rag_ingest`: Substance JSON ingestion

## How It Works

### Step 1: Extract Root Section
When a chunk is received, the enricher determines its root section:

```python
# Priority order for determining root section:
1. Explicit "root_section" in metadata
2. "hierarchy.parent_section" in metadata
3. Section field value (if it's "root")
4. Fallback to "root"
```

### Step 2: Identify Parent Identity
Parent is identified by `(document_id, root_section)`:

```python
parent_identity = ParentIdentity(
    document_id=chunk.document_id,
    root_section=root_section
)
```

### Step 3: Reconstruct Parent Context
Load all chunks with same parent identity:

```python
# Get all chunks for this document
all_chunks = db.get_documents_by_substance(parent_identity.document_id)

# Filter by root section
parent_chunks = [
    c for c in all_chunks 
    if extract_root_section(c) == parent_identity.root_section
]

# Build parent context from parent_chunks
```

### Step 4: Enrich Observations
Augment child chunks with parent information:

```python
enriched = {
    "chunk": {
        "document_id": "...",
        "section": "...",
        "text": "...",
        ...
    },
    "parent_context": {
        "num_chunks": 5,
        "sections_included": ["root", "names", "codes"],
        "text_parts": [...],  # Parts from all parent chunks
        "metadata_unified": {...}
    },
    "parent_text_summary": "Aggregated text from parent..."
}
```

## Performance Considerations

### Efficiency
- **No schema changes**: Uses existing chunk structure
- **No reindexing**: Existing vectors remain valid
- **On-demand reconstruction**: Parent context built only when needed
- **Caching in results**: Parent contexts cached across multiple results

### When to Add Dedicated Storage
Measure performance and consider dedicated parent storage if:
1. Parent reconstruction is frequently accessed
2. Document sizes are very large (>1000 chunks)
3. Latency requirements are strict (<100ms)

## Data Flow

```
User Query
    ↓
Embedding Service
    ↓
Vector DB Similarity Search → [DBQueryResult] (child chunks)
    ↓
Parent Context Enricher
    ↓
For each result chunk:
    1. Extract root section
    2. Build parent identity
    3. Load all parent chunks
    4. Reconstruct parent context
    5. Enrich child chunk
    ↓
Enhanced Results with Parent Context
    ↓
Format and Return to User
```

## Usage Examples

### Example 1: RAG Query with Parent Context
```python
# Client code
results = await rag_query_with_parent_context(
    query="What is the mechanism of action?",
    top_k=5,
    include_parent_text=True
)
```

### Example 2: Get Parent Context for Specific Chunk
```python
# Client code
parent_context = await get_parent_context(
    chunk_id="root_uuid:12345678-..."
)
```

### Example 3: Direct Service Usage
```python
# In application code
enricher = runtime.vector_db.parent_enricher

# Get parent for a chunk
parent_identity = enricher.get_parent_identity(chunk)
parent_context = enricher.reconstruct_parent_context(parent_identity)

# Enrich multiple results
enriched_results = enricher.enrich_search_results(
    search_results,
    include_parent_text=True
)
```

## Testing

Comprehensive test suite in `tests/test_parent_child_retrieval.py`:

```bash
# Run all parent-child tests
python -m pytest tests/test_parent_child_retrieval.py -v

# Run specific test class
python -m pytest tests/test_parent_child_retrieval.py::TestParentContextEnricher -v

# Run with coverage
python -m pytest tests/test_parent_child_retrieval.py --cov=app.services.parent_child_retrieval
```

### Test Coverage
- `ParentIdentity`: Creation, equality, hashing
- `ParentContextEnricher`: 
  - Root section extraction (from metadata, hierarchy, section, fallback)
  - Parent identity extraction
  - Parent context reconstruction (success, exclusion, empty)
  - Chunk enrichment
  - Batch enrichment of search results
  - Integration workflows

## Future Enhancements

### Option 1: Dedicated Parent Storage
If performance measurements show need:
```sql
CREATE TABLE parent_chunks (
    id SERIAL PRIMARY KEY,
    parent_identity_hash TEXT,
    document_id UUID,
    root_section TEXT,
    aggregated_text TEXT,
    metadata JSONB,
    created_at TIMESTAMP,
    UNIQUE(document_id, root_section)
);
```

### Option 2: Hierarchical Storage
Add explicit parent pointers:
```python
class VectorDocument(Base):
    parent_chunk_id: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    hierarchy_level: Mapped[int] = mapped_column(Integer, default=0)
```

### Option 3: Caching Layer
Add Redis/Memcached for frequently accessed parent contexts:
```python
cache_key = f"parent:{document_id}:{root_section}"
parent_context = cache.get(cache_key) or reconstruct_from_db()
```

## Configuration

The implementation uses default parameters that can be customized:

```python
# In enrichment calls:
enricher.enrich_search_results(
    results,
    include_parent_text=True,        # Include text summaries
    parent_text_limit=1000,           # Max chars per parent text
)

enricher.enrich_chunk_with_parent(
    chunk,
    parent_context=None,              # Pre-compute if available
    include_parent_text=True,
    parent_text_limit=1000,
)
```

## Error Handling

The implementation gracefully handles:
- **Missing chunks**: Returns None for parent context if no chunks found
- **Empty documents**: Returns empty parent identity list
- **Malformed metadata**: Falls back to sensible defaults
- **Section extraction failures**: Uses "root" as fallback

## Integration Points

1. **Vector Database Service**: `vector_db.parent_enricher` property
2. **MCP Tools**: `rag_query_with_parent_context`, `get_parent_context`
3. **Backend Queries**: Optimized queries for filtered retrieval
4. **Existing RAG Flow**: Non-breaking additions to existing tools

## Migration Path

No migration required:
1. All changes are additive
2. Existing `rag_query` tool unchanged
3. Existing database schema unchanged
4. New tools available immediately upon deployment
5. Can be enabled gradually for specific use cases
