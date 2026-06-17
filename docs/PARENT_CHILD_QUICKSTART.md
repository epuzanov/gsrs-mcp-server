# Parent-Child Retrieval: Quick Start Guide

## Overview

The parent-child retrieval system automatically enriches retrieval results with context from related chunks. It reconstructs parent contexts virtually from existing data without requiring schema changes or reindexing.

## Prerequisites

- Ingested GSRS substance documents in the vector database
- Working embedding service
- Vector database backend (pgvector or ChromaDB)

## Using the New Tools

### Option 1: Query with Parent Context Enrichment

Use `rag_query` to automatically enrich search results:

```
Tool: rag_query

Parameters:
- query (required): "What is the mechanism of action for aspirin?"
- top_k (optional, default=8): 5
- include_parent_text (optional, default=true): true
- parent_text_limit (optional, default=1000): 500
- filters (optional): '{"document_id": "550e8400-e29b-41d4-a716-446655440000"}'

Example Request:
{
  "query": "chemical structure and composition",
  "top_k": 5,
  "include_parent_text": true,
  "parent_text_limit": 800
}

Example Response:
Found 3 local RAG result(s) with parent context for "chemical structure and composition":

1. Score: 0.96 | Section: structure
   Substance UUID: 550e8400-e29b-41d4-a716-446655440000
   Chunk: root_uuid:12345678-...
   Text: The molecular structure consists of a benzene ring with an acetyl group...
   Parent Context: 6 chunks in root, names, codes, structure, mechanisms, references
   Parent Summary: [Acetylsalicylic acid overview...] [Common names include aspirin...]

2. Score: 0.89 | Section: codes
   Substance UUID: 550e8400-e29b-41d4-a716-446655440000
   Chunk: root_uuid:87654321-...
   Text: UNII: R16CO5Y76E, CAS: 50-78-2...
   Parent Context: 6 chunks in root, names, codes, structure, mechanisms, references
   Parent Summary: [Acetylsalicylic acid overview...] [Common regulatory codes...]

3. Score: 0.82 | Section: mechanisms
   ...
```

### Option 2: Explore Parent Context for a Specific Chunk

Use `get_parent_context` to examine all context chunks for a specific result:

```
Tool: get_parent_context

Parameters:
- chunk_id (required): "root_uuid:12345678-..."

Example Request:
{
  "chunk_id": "root_uuid:12345678-abcd-ef01-2345-6789abcdef01"
}

Example Response:
Parent Context for Chunk: root_uuid:12345678-abcd-ef01-2345-6789abcdef01
Document: 550e8400-e29b-41d4-a716-446655440000
Root Section: root
Num Chunks in Parent: 6
Sections Included: root, names, codes, structure, mechanisms, references

Text Parts:
1. [root] Acetylsalicylic Acid is an acetylated salicylic acid derivative... 
2. [names] Common names include Aspirin, 2-acetoxybenzoic acid, and acetylsalicylic acid...
3. [codes] UNII: R16CO5Y76E, CAS: 50-78-2, FDA Approval: Yes...
4. [structure] Molecular structure: C9H8O4, contains benzene ring with acetyl group...
5. [mechanisms] Irreversible inhibitor of cyclooxygenase (COX) enzymes...
6. [references] WHO List: J01BA51, FDA Orange Book entries...

Unified Metadata:
- display_name: Acetylsalicylic Acid
- substance_class: Chemical
- substance_type: Small Molecule
- regulatory_status: Approved
- common_alternate_names: Aspirin, Acetylsalicylic Acid, 2-Acetoxybenzoic Acid
```

## Understanding Parent Context

### What is a Parent?
A parent groups chunks belonging to the same document and root section:
- **Document**: Identified by `document_id` (substance UUID)
- **Root Section**: The topmost section in the document (usually "root")
- **Children**: All chunks with the same parent identity

### Example Document Structure
```
Document: Acetylsalicylic Acid (UUID: 550e8400-...)

Chunks (Parent: document_id=550e8400-..., root_section=root):
├── root_overview          [main substance definition]
├── names_aspirin          [common names]
├── names_acetyl           [chemical names]
├── codes_unii             [UNII code]
├── codes_cas              [CAS number]
├── codes_fda_approval     [FDA approval ID]
├── structure_molecular    [molecular structure info]
├── structure_properties   [chemical properties]
├── mechanisms_cox         [mechanism of action]
├── mechanisms_absorption  [pharmacokinetics]
├── references_who_list    [WHO classification]
└── references_fda_orange  [FDA references]

When you retrieve any chunk from this group, the parent enricher
automatically provides context from ALL chunks in the parent.
```

## How Root Section is Determined

The enricher uses this priority order:

1. **Explicit `root_section` in metadata**
   ```python
   metadata_json = {"root_section": "compound"}
   ```
   Uses: `"compound"`

2. **Hierarchy parent section in metadata**
   ```python
   metadata_json = {"hierarchy": {"parent_section": "compound"}}
   ```
   Uses: `"compound"`

3. **Section field value (if it's "root")**
   ```python
   section = "root"
   metadata_json = {}
   ```
   Uses: `"root"`

4. **Section field (as-is)**
   ```python
   section = "names"
   metadata_json = {}
   ```
   Uses: `"names"`

5. **Fallback to "root"**
   ```python
   section = ""
   metadata_json = {}
   ```
   Uses: `"root"`

## Integration Examples

### In Claude or LLM Context
When using parent context results in a multi-turn conversation:

```
User: "What are the side effects of aspirin?"

Assistant uses rag_query:
1. Searches for "side effects aspirin"
2. Gets child chunks about side effects with parent context
3. Parent context automatically includes:
   - Substance overview
   - Chemical structure
   - Mechanisms of action
   - Related references

This gives the LLM fuller context: the side effects make more sense
when the LLM knows the substance's mechanism of action.
```

### In Retrieval Workflows
```
User Query
    ↓
Embed and search
    ↓
Get top-k results [children]
    ↓
Enrich with parent context → [enriched_results]
    ↓
Format and return to LLM
```

## Performance Tips

### Query Parameters
- **`top_k`**: Start with 5-8 for most queries, increase for broader context
- **`include_parent_text`**: Set to `true` for better context, `false` for minimal data
- **`parent_text_limit`**: Use 500-1000 for balance, increase for detailed context
- **`filters`**: Filter by `document_id` if you know the substance UUID

### Example Queries

**For quick, focused answers:**
```
rag_query(
  query="aspirin dosage",
  top_k=3,
  parent_text_limit=500
)
```

**For comprehensive understanding:**
```
rag_query(
  query="mechanism of action",
  top_k=8,
  parent_text_limit=1000,
  include_parent_text=true
)
```

**For specific document:**
```
rag_query(
  query="side effects",
  filters='{"document_id": "550e8400-e29b-41d4-a716-446655440000"}',
  top_k=5
)
```

## Troubleshooting

### No parent context found
**Symptom**: Result shows chunk but no parent information

**Causes**:
1. Chunk has no related chunks in same parent
2. All parent chunks are excluded
3. Document hasn't been fully ingested

**Solutions**:
- Check if document is fully ingested with `statistics` tool
- Try `get_parent_context` on the chunk directly
- Verify document structure with `gsrs_get_substance`

### Incomplete parent context
**Symptom**: Parent context shows fewer chunks than expected

**Causes**:
1. Parent reconstruction is excluding sections
2. Chunks with different root_section values
3. Ingestion was partial

**Solutions**:
- Check chunk metadata with `get_parent_context`
- Re-ingest the full substance document
- Verify root_section detection with diagnostic queries

### Slow parent context enrichment
**Symptom**: `rag_query` returns slowly

**Causes**:
1. Very large documents (100s of chunks)
2. Many unique parents in results
3. Network latency to database

**Solutions**:
- Reduce `parent_text_limit` to decrease data transfer
- Use `include_parent_text=false` for initial retrieval
- Filter by `document_id` to reduce result diversity
- Consider metrics-driven optimization (see docs)

## Next Steps

1. **Try it out**: Use `rag_query` in your queries
2. **Compare results**: Check improvement vs `rag_query_chunks`
3. **Measure impact**: Track LLM answer quality improvements
4. **Provide feedback**: Report performance metrics and use cases
5. **Optimize**: Based on metrics, consider dedicated parent storage

## Related Documentation

- `PARENT_CHILD_IMPLEMENTATION.md` - Implementation details
- `docs/PARENT_CHILD_RETRIEVAL.md` - Full technical documentation
- `app/services/parent_child_retrieval.py` - Source code
- `tests/test_parent_child_retrieval.py` - Test examples

## API Reference

### rag_query

```
POST /mcp/tools/rag_query

Parameters:
- query (str, required): Search query text
- top_k (int, default=8): Number of results
- include_parent_text (bool, default=true): Include parent summaries
- parent_text_limit (int, default=1000): Max chars for parent text
- filters (str, default=""): JSON filter object

Returns:
- Formatted string with enriched results
- Each result includes: score, section, chunk info, parent context
```

### rag_query_chunks

```
POST /mcp/tools/rag_query_chunks

Parameters:
- query (str, required): Search query text
- top_k (int, default=8): Number of raw chunk results
- filters (str, default=""): JSON filter object

Returns:
- Formatted string with raw chunk results (no parent context)
```

### get_parent_context

```
POST /mcp/tools/get_parent_context

Parameters:
- chunk_id (str, required): Chunk ID from search results

Returns:
- Formatted string with parent context details
- Includes: parent identity, num chunks, sections, text parts, metadata
```

## Questions?

Refer to the full documentation in `docs/PARENT_CHILD_RETRIEVAL.md` for:
- Architecture details
- Performance characteristics
- Future enhancement options
- Data structure specifications
