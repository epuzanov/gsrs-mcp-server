# Data Loading Guide

How to load GSRS substance data into the MCP Server.

## Overview

The GSRS MCP Server supports multiple methods for loading substance data:

1. **API Endpoints** - Single or batch ingest via REST API
2. **Load Script** - Command-line tool for bulk loading from files or GSRS server
3. **Direct Download** - Fetch substances directly from GSRS NCATS API

## Supported Formats

| Format | Description | Tool |
|--------|-------------|------|
| JSON | Single substance document | API `/ingest` |
| JSON Array | Multiple substances | API `/ingest/batch` |
| .gsrs (JSONL.gz) | Compressed JSONL with 2 leading tabs | `load_data.py` |

## Loading from Files

### Single JSON File

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d @substance.json
```

### Multiple JSON Files

```bash
for file in substances/*.json; do
    curl -X POST http://localhost:8000/ingest \
        -H "Content-Type: application/json" \
        -d @"$file"
done
```

### Batch API

```bash
curl -X POST http://localhost:8000/ingest/batch \
  -H "Content-Type: application/json" \
  -d '{"substances": [...]}'
```

## Load Script

The `load_data.py` script supports multiple input sources:

### From .gsrs File

```bash
python scripts/load_data.py data/substances.gsrs \
  --batch-size 100 \
  --api-url http://localhost:8000
```

### From UUID List

Load specific substances from GSRS server:

```bash
python scripts/load_data.py \
  --uuids 0103a288-6eb6-4ced-b13a-849cd7edf028,80edf0eb-b6c5-4a9a-adde-28c7254046d9
```

### Download All Substances

Load all substances from GSRS server (may take hours):

```bash
python scripts/load_data.py --all --max-results 10000
```

### Dry Run

Parse/download without uploading:

```bash
python scripts/load_data.py data.gsrs --dry-run
```

### Script Options

| Option | Default | Description |
|--------|---------|-------------|
| `file` | - | Path to .gsrs file |
| `--uuids` | - | Comma-separated UUIDs to load from GSRS |
| `--all` | - | Load all substances from GSRS server |
| `--batch-size` | 100 | Substances per batch |
| `--api-url` | `http://localhost:8000` | MCP Server URL |
| `--dry-run` | - | Parse/download without uploading |
| `--max-results` | 10000 | Max substances when using `--all` |

### Example Output

```
2026-03-22 23:56:43 - INFO - Loading 2 substances from GSRS API...
2026-03-22 23:56:44 - INFO - Downloading 2 substances from GSRS API...
2026-03-22 23:56:45 - INFO - Downloaded 2/2 substances

============================================================
LOAD SUMMARY
============================================================
Substances downloaded: 2/2
Total substances processed: 2
Successful: 2
Failed: 0
Total chunks created: 45
============================================================
```

## .gsrs File Format

The `.gsrs` format is JSONL (JSON Lines) compressed with gzip, where each line starts with two tab characters.

### Creating .gsrs Files

```python
import gzip
import json

substances = [
    {"uuid": "uuid-1", "substanceClass": "chemical", ...},
    {"uuid": "uuid-2", "substanceClass": "protein", ...}
]

with gzip.open('substances.gsrs', 'wt', encoding='utf-8') as f:
    for substance in substances:
        f.write('\t\t' + json.dumps(substance) + '\n')
```

### Reading .gsrs Files

```python
import gzip
import json

with gzip.open('substances.gsrs', 'rt') as f:
    for line in f:
        substance = json.loads(line.lstrip('\t\t'))
        # Process substance
```

## Downloading Sample Data

### Manual Download

```bash
# Single substance
curl -o substance.json \
  "https://gsrs.ncats.nih.gov/api/v1/substances(0103a288-6eb6-4ced-b13a-849cd7edf028)?view=full"
```

## GSRS Substance JSON Format

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `uuid` | string | Unique substance identifier |
| `substanceClass` | string | Type determinant |

### Example Structure

```json
{
  "uuid": "0103a288-6eb6-4ced-b13a-849cd7edf028",
  "substanceClass": "chemical",
  "status": "approved",
  "approvalID": "WK2XYI10QM",
  "names": [
    {"name": "Ibuprofen", "type": "INN", "preferred": true}
  ],
  "codes": [
    {"code": "15687-27-1", "codeSystem": "CAS", "type": "PRIMARY"}
  ],
  "structure": {
    "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "formula": "C13H18O2"
  }
}
```

### Substance Classes

| Class | Description |
|-------|-------------|
| `concept` | Base substance class for concepts |
| `chemical` | Chemical substances |
| `protein` | Proteins and peptides |
| `nucleicAcid` | DNA/RNA sequences |
| `polymer` | Polymers |
| `mixture` | Mixtures of multiple components |
| `structurallyDiverse` | Natural products, organisms |
| `specifiedSubstanceG1` | Specified substances |

## Chunking

Each substance is automatically chunked based on its structure:

### Overview Chunk

Every substance gets an overview chunk with key metadata:

```
Substance Class: chemical | UUID: ... | Definition Type: PRIMARY | Status: approved | Names: 5 entries | Codes: 10 entries
```

### Section Chunks

Common sections are chunked:

| Section | Chunk Paths |
|---------|-------------|
| names | `root_names_inn`, `root_names_brand_name`, etc. |
| codes | `root_codes_cas_XXX`, `root_codes_fda_unii_XXX` |
| properties | `root_properties_molecular_weight`, etc. |
| relationships | `root_relationships_metabolite_parent`, etc. |

### Class-Specific Chunks

| Class | Specific Chunks |
|-------|-----------------|
| chemical | `structure`, `moieties_0`, `moieties_1`, ... |
| protein | `protein_overview`, `protein_subunit_1`, `protein_glycosylation`, ... |
| nucleicAcid | `nucleic_acid_overview`, `nucleic_acid_subunit_1`, ... |
| polymer | `polymer_classification`, `polymer_monomer_0`, ... |
| mixture | `mixture_component_0`, `mixture_parent` |
| structurallyDiverse | `structurally_diverse_source` |

See [Chunking Guide](guides/chunking.md) for details.

## ALTERNATIVE Definitions

For substances with `definitionType: "ALTERNATIVE"`:

- The `substance_uuid` is taken from the `refuuid` of the `SUB_ALTERNATE->SUBSTANCE` relationship
- This ensures PRIMARY and ALTERNATIVE definitions share the same UUID
- The overview chunk includes "Alternative Definition for: {Name}"

## Bulk Loading Best Practices

### 1. Use Appropriate Batch Sizes

```bash
# Small datasets (< 100 substances)
--batch-size 50

# Large datasets (> 1000 substances)
--batch-size 200
```

### 2. Monitor Progress

```bash
# Watch statistics during load
watch -n 5 'curl -s http://localhost:8000/statistics'
```

### 3. Check API Health

```bash
# Before loading
curl http://localhost:8000/health

# Expected response
{"status": "healthy", "database_connected": true, "model_loaded": true}
```

### 4. Handle Errors

```bash
# Log output
python scripts/load_data.py data.gsrs 2>&1 | tee load.log

# Check for failures
grep "Failed" load.log
```

## Verifying Loaded Data

### Check Statistics

```bash
curl http://localhost:8000/statistics
```

Response:
```json
{
  "total_chunks": 1250,
  "total_substances": 50
}
```

### Test Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "aspirin CAS code", "top_k": 3}'
```

## Troubleshooting

### API Not Available

```bash
# Check if server is running
curl http://localhost:8000/health

# Start server if needed
uvicorn app.main:app --reload
```

### Database Connection Failed

```bash
# Check DATABASE_URL
echo $DATABASE_URL

# For ChromaDB (default)
DATABASE_URL=chroma://./chroma_data/chunks

# For PostgreSQL
DATABASE_URL=postgresql://user:pass@localhost:5432/gsrs_rag
```

### No Chunks Created

```bash
# Check substance structure
python -c "import json; print(json.load(open('substance.json')))"

# Verify required fields: uuid, substanceClass
```

### Batch Load Fails

```bash
# Try smaller batch size
python scripts/load_data.py data.gsrs --batch-size 10

# Check API logs
docker-compose logs gsrs-rag-gateway
```

### GSRS API Rate Limiting

When downloading many substances:

```bash
# Limit concurrent requests
# The script handles this automatically with batch processing

# Add delays between batches
# Modify script if needed
```

### Out of Memory

```bash
# Reduce batch size
--batch-size 25

# For Docker, increase memory limit
# Edit Docker Desktop settings
```

## Related Documentation

- [Quick Start](quickstart.md) - Get started quickly
- [Chunking Guide](guides/chunking.md) - Detailed chunking documentation
- [API Reference](api-reference.md) - API endpoints
- [Configuration](configuration.md) - Configuration options
