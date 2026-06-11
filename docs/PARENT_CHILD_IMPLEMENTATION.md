# Parent-Child Retrieval: Implementation Summary

## What Was Implemented

A **virtual parent reconstruction** system for parent-child retrieval that:

✅ **Requires no schema modifications** - Uses existing chunk data
✅ **Requires no reindexing** - Works with existing embeddings  
✅ **Requires no migrations** - Fully backward compatible
✅ **Delivers parent context** - Automatically enriches search results
✅ **Defers dedicated storage** - Can be added later based on metrics

## Key Features

### 1. Virtual Parent Identity
Parents are identified by `(document_id, root_section)`:
- Each substance has chunks across different sections (root, names, codes, etc.)
- Chunks sharing the same root_section form a parent group
- Parent context reconstructed on-demand from all chunks in the group

### 2. Automatic Root Section Detection
Priority order for determining root section:
1. Explicit `root_section` in metadata
2. `hierarchy.parent_section` from metadata
3. Section name (if "root")
4. Fallback to "root"

### 3. Enrichment Pipeline
```
Search Results
    ↓
Extract parent identity from each chunk
    ↓
Load all chunks with same parent identity
    ↓
Build unified parent context
    ↓
Augment child chunk with parent info
    ↓
Return enriched results
```

## Files Created/Modified

### New Files
- `app/services/parent_child_retrieval.py` - Core parent context enricher
- `tests/test_parent_child_retrieval.py` - Comprehensive test suite
- `docs/PARENT_CHILD_RETRIEVAL.md` - Detailed documentation

### Modified Files
- `app/db/base.py` - Added `get_documents_by_section_and_substance()` interface
- `app/db/backends/pgvector.py` - Optimized PostgreSQL query for section filtering
- `app/db/backends/chroma.py` - Chroma backend implementation
- `app/services/vector_database.py` - Service wrapper and parent enricher property
- `app/main.py` - Added 2 new MCP tools + updated instructions

## New MCP Tools

### 1. `rag_query_with_parent_context`
Performs RAG query with automatic parent context enrichment.

**Parameters:**
- `query` (str) - Search query text
- `top_k` (int, default=8) - Number of results to return
- `include_parent_text` (bool, default=True) - Include parent text summaries
- `parent_text_limit` (int, default=1000) - Max chars for parent text
- `filters` (str) - Optional JSON filters

**Returns:** Search results with parent context

```
Found 3 local RAG result(s) with parent context for "mechanism of action":

1. Score: 0.95 | Section: mechanisms
   Substance UUID: <uuid>
   Chunk: root_uuid:...
   Text: [512 chars of chunk content]
   Parent Context: 5 chunks in root, names, codes, mechanisms, references
   Parent Summary: [aggregated text from parent chunks]

2. Score: 0.87 | Section: codes
   ...
```

### 2. `get_parent_context`
Retrieve parent context for a specific chunk.

**Parameters:**
- `chunk_id` (str) - The chunk ID to get parent for

**Returns:** Parent context details

```
Parent Context for Chunk: root_uuid:12345678-...
Document: <substance-uuid>
Root Section: root
Num Chunks in Parent: 5
Sections Included: root, names, codes, mechanisms, references

Text Parts:
1. [root] Acetylsalicylic Acid is an acetylated salicylic acid derivative...
2. [names] Common names: Aspirin, 2-acetoxybenzoic acid...
3. [codes] UNII: R16CO5Y76E, CAS: 50-78-2...
...

Unified Metadata:
- display_name: Acetylsalicylic Acid
- substance_class: Chemical
- approval_status: Approved
...
```

## Core Classes

### `ParentIdentity`
```python
@dataclass
class ParentIdentity:
    document_id: UUID      # Substance UUID
    root_section: str      # Top-level section

    def __hash__(self):
        return hash((self.document_id, self.root_section))
```

### `ParentContextEnricher`
```python
class ParentContextEnricher:
    def extract_root_section(chunk: VectorDocument) -> str:
        """Extract root section from chunk"""
    
    def get_parent_identity(chunk: VectorDocument) -> ParentIdentity:
        """Get parent identity for chunk"""
    
    def reconstruct_parent_context(
        parent_identity: ParentIdentity
    ) -> Optional[Dict[str, Any]]:
        """Load and aggregate all parent chunks"""
    
    def enrich_chunk_with_parent(
        chunk: VectorDocument
    ) -> Dict[str, Any]:
        """Add parent context to chunk"""
    
    def enrich_search_results(
        results: List[DBQueryResult]
    ) -> List[Dict[str, Any]]:
        """Enrich multiple results, caching parent contexts"""
```

## Data Structure

### Parent Context Format
```python
{
    "parent_identity": {
        "document_id": "550e8400-e29b-41d4-a716-446655440000",
        "root_section": "root"
    },
    "num_chunks": 5,
    "sections_included": ["root", "names", "codes", "mechanisms"],
    "text_parts": [
        {
            "section": "root",
            "text": "[truncated to 500 chars]",
            "source_url": "..."
        },
        ...
    ],
    "metadata_unified": {
        "display_name": "Acetylsalicylic Acid",
        "substance_class": "Chemical",
        ...
    }
}
```

### Enriched Result Format
```python
{
    "chunk": {
        "document_id": "...",
        "section": "names",
        "chunk_id": "root_uuid:...",
        "text": "Aspirin is a commonly used name...",
        "source_url": "...",
        "metadata": {...}
    },
    "score": 0.95,
    "parent_context": {
        "parent_identity": {...},
        "num_chunks": 5,
        "sections_included": [...],
        "text_parts": [...]
    },
    "parent_text_summary": "Aggregated text from parent..."
}
```

## Usage Examples

### Example 1: Search with Parent Context
```python
# Use the MCP tool
results = await rag_query_with_parent_context(
    query="What is the chemical structure?",
    top_k=5,
    include_parent_text=True,
    parent_text_limit=1000
)
```

### Example 2: Explore Parent for Chunk
```python
# Get parent context for specific chunk
parent = await get_parent_context(
    chunk_id="root_uuid:12345678-..."
)
```

### Example 3: Direct Service Usage
```python
# In Python code
enricher = runtime.vector_db.parent_enricher

# Single chunk enrichment
enriched = enricher.enrich_chunk_with_parent(chunk)

# Batch enrichment with caching
enriched_results = enricher.enrich_search_results(
    search_results,
    include_parent_text=True
)
```

## Performance Characteristics

### Time Complexity (per result)
- Extract root section: O(1)
- Get parent identity: O(1)  
- Reconstruct parent: O(n) where n = chunks in document
- Enrich chunk: O(1)
- **Batch enrichment**: O(m) amortized where m = unique parents

### Space Complexity
- Parent context: O(n) where n = chunks in parent
- Cached contexts: O(unique_parents * avg_parent_size)

### Database Queries
- Single query per unique parent identity (cached)
- Uses indexed `(document_id, section)` lookup
- No full-text search or vector operations needed

## Backward Compatibility

✅ **Fully backward compatible:**
- Existing `rag_query` tool unchanged
- Existing database schema untouched
- New tools are additive
- All changes are opt-in

## Testing

Complete test suite covering:

```bash
# Run all parent-child tests
python -m pytest tests/test_parent_child_retrieval.py -v

# Test categories:
- ParentIdentity: creation, equality, hashing
- Root section extraction: metadata, hierarchy, section, fallback
- Parent context reconstruction: success, exclusion, empty
- Chunk enrichment: single and batch
- Integration: full workflow
```

## Future Enhancements

### Option 1: Metrics-Driven Decision
Monitor these metrics to decide if dedicated storage needed:
- Average parent reconstruction time
- Number of parent reconstructions per second
- Cache hit rate
- User latency requirements

### Option 2: Dedicated Parent Storage
If metrics show need, can add:
```sql
CREATE TABLE parent_chunks (
    document_id UUID,
    root_section TEXT,
    aggregated_text TEXT,
    metadata JSONB,
    PRIMARY KEY (document_id, root_section)
);
```
- No migration needed for existing child chunks
- New parent storage updated async
- Existing tools continue working

### Option 3: Redis Cache Layer
For high-throughput scenarios:
```python
cache_key = f"parent:{doc_id}:{root_section}"
parent = redis.get(cache_key) or reconstruct()
```

## Benefits Summary

| Feature | Benefit |
|---------|---------|
| Virtual reconstruction | No schema changes, migrations, or reindexing |
| On-demand | Parent context only built when needed |
| Automatic enrichment | Search results include parent context |
| Caching | Multiple results share cached parents |
| Backward compatible | Existing tools and data unchanged |
| Deferrable | Dedicated storage can be added later |
| Measurable | Clear metrics for optimization decisions |

## Next Steps

1. **Deploy** - New tools available immediately
2. **Monitor** - Track parent reconstruction metrics
3. **Optimize** - Use metrics to decide on dedicated storage
4. **Measure** - Assess LLM response quality improvements
5. **Iterate** - Refine based on user feedback

## References

- Implementation: `app/services/parent_child_retrieval.py`
- Tests: `tests/test_parent_child_retrieval.py`
- Documentation: `docs/PARENT_CHILD_RETRIEVAL.md`
- MCP Tools: `app/main.py` (search for `rag_query_with_parent_context`)
