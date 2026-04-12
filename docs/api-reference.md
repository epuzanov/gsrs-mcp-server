# API Reference

Complete API reference for the GSRS MCP Server.

## Base URL

```
http://localhost:8000
```

## Authentication

Most endpoints require **HTTP Basic Authentication**. Include credentials using the `-u` flag in curl or the `auth` parameter in Python.

```bash
# Default credentials (change in production!)
Username: admin
Password: admin123
```

See [Authentication Guide](authentication.md) for details.

---

## Endpoints

### Health Check

Check the health status of the service.

```http
GET /health
```

**Authentication:** Not required

**Response:**

```json
{
  "status": "healthy",
  "database_connected": true,
  "statistics": {
    "total_chunks": 150,
    "total_substances": 17
  }
}
```

---

### Ingest Substance

Ingest a single GSRS Substance document.

```http
POST /ingest
Content-Type: application/json
Authorization: Basic <credentials>
```

**Authentication:** Required

**Request Body:**

```json
{
  "substance": {
    "uuid": "0103a288-6eb6-4ced-b13a-849cd7edf028",
    "substanceClass": "chemical",
    "names": [
      {
        "name": "Aspirin",
        "type": "COMMON",
        "preferred": true
      }
    ],
    "codes": [
      {"code": "50-78-2", "codeSystem": "CAS"}
    ],
    "structure": {
      "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"
    }
  }
}
```

**Response:**

```json
{
  "substance_uuid": "0103a288-6eb6-4ced-b13a-849cd7edf028",
  "chunks_created": 25,
  "element_paths": [
    "root_names_0_name",
    "root_codes_0_code",
    "root_codes_0_codeSystem"
  ]
}
```

**Status Codes:**
- `200`: Success
- `400`: Invalid substance JSON
- `401`: Authentication required

---

### Batch Ingest

Ingest multiple substances in one request.

```http
POST /ingest/batch
Content-Type: application/json
Authorization: Basic <credentials>
```

**Authentication:** Required

**Request Body:**

```json
{
  "substances": [
    {
      "uuid": "0103a288-6eb6-4ced-b13a-849cd7edf028",
      "substanceClass": "chemical",
      "names": [{"name": "Aspirin", "type": "COMMON"}]
    },
    {
      "uuid": "80edf0eb-b6c5-4a9a-adde-28c7254046d9",
      "substanceClass": "chemical",
      "names": [{"name": "Ibuprofen", "type": "COMMON"}]
    }
  ]
}
```

**Response:**

```json
{
  "total_substances": 2,
  "total_chunks": 50,
  "successful": 2,
  "failed": 0,
  "errors": []
}
```

**Status Codes:**
- `200`: Success
- `400`: Invalid request body
- `401`: Authentication required

---

### Query

Perform semantic search on substance chunks.

```http
POST /query
Content-Type: application/json
Authorization: Basic <credentials>  # Optional but recommended
```

**Authentication:** Optional

**Request Body:**

```json
{
  "query": "CAS code for Aspirin",
  "top_k": 5,
  "filters": {}
}
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | - | Search query text |
| `top_k` | integer | No | 5 | Number of results (1-100) |
| `filters` | object | No | `{}` | Additional metadata filters |

**Response:**

```json
{
  "query": "CAS code for Aspirin",
  "results": [
    {
      "chunk_id": "root_codes_0_code",
      "document_id": "0103a288-6eb6-4ced-b13a-849cd7edf028",
      "section": "codes",
      "text": "code: 50-78-2",
      "similarity_score": 0.892,
      "chunk_metadata": {
        "codeSystem": "CAS",
        "chunk_type": "code"
      }
    }
  ],
  "total_results": 1
}
```

**Status Codes:**
- `200`: Success
- `400`: Invalid query parameters
- `401`: Authentication required (if enabled)

---

### Delete Substance

Delete all chunks for a specific substance.

```http
DELETE /substances/{substance_uuid}
Authorization: Basic <credentials>
```

**Authentication:** Required

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `substance_uuid` | UUID | The substance UUID to delete |

**Response:**

```json
{
  "substance_uuid": "0103a288-6eb6-4ced-b13a-849cd7edf028",
  "chunks_deleted": 25
}
```

**Status Codes:**
- `200`: Success
- `404`: Substance not found
- `401`: Authentication required

---

### Get Models

Get information about available embedding models.

```http
GET /models
```

**Authentication:** Not required

**Response:**

```json
{
  "models": [
    {
      "provider": "openai",
      "model": "text-embedding-3-small",
      "dimension": 1536,
      "base_url": "https://api.openai.com/v1"
    }
  ],
  "current_model": "text-embedding-3-small"
}
```

---

### Get Substance Classes

Get list of available GSRS substance classes in the database.

```http
GET /substance-classes
```

**Authentication:** Not required

**Response:**

```json
{
  "substance_classes": [
    "chemical",
    "protein",
    "nucleicAcid",
    "polymer",
    "mixture",
    "structurallyDiverse",
    "concept",
    "specifiedSubstanceG1"
  ]
}
```

---

### Get Statistics

Get database statistics.

```http
GET /statistics
Authorization: Basic <credentials>
```

**Authentication:** Required

**Response:**

```json
{
  "total_chunks": 150,
  "total_substances": 17
}
```

---

## Error Responses

### 400 Bad Request

```json
{
  "detail": "Invalid request body: missing required field 'substance'"
}
```

### 401 Unauthorized

```json
{
  "detail": "Could not validate credentials"
}
```

### 404 Not Found

```json
{
  "detail": "Substance not found: 0103a288-6eb6-4ced-b13a-849cd7edf028"
}
```

### 500 Internal Server Error

```json
{
  "detail": "Internal server error: database connection failed"
}
```

---

## OpenAPI Schema

The complete OpenAPI schema is available at:

```
http://localhost:8000/openapi.json
```

## Interactive Documentation

Swagger UI is available at:

```
http://localhost:8000/docs
```

ReDoc is available at:

```
http://localhost:8000/redoc
```

---

## Code Examples

### Python (httpx)

```python
import httpx

BASE_URL = "http://localhost:8000"
AUTH = ("admin", "admin123")

# Health check (no auth required)
response = httpx.get(f"{BASE_URL}/health")
print(response.json())

# Query (auth optional)
response = httpx.post(
    f"{BASE_URL}/query",
    auth=AUTH,
    json={"query": "Aspirin CAS code", "top_k": 5}
)
results = response.json()["results"]
print(f"Found {len(results)} results")

# Ingest (auth required)
response = httpx.post(
    f"{BASE_URL}/ingest",
    auth=AUTH,
    json={"substance": substance_data}
)
print(f"Created {response.json()['chunks_created']} chunks")

# Delete (auth required)
response = httpx.delete(
    f"{BASE_URL}/substances/0103a288-6eb6-4ced-b13a-849cd7edf028",
    auth=AUTH
)
print(f"Deleted {response.json()['chunks_deleted']} chunks")
```

### Python (requests)

```python
import requests
from requests.auth import HTTPBasicAuth

BASE_URL = "http://localhost:8000"
AUTH = HTTPBasicAuth("admin", "admin123")

# Query
response = requests.post(
    f"{BASE_URL}/query",
    auth=AUTH,
    json={"query": "Aspirin", "top_k": 5}
)
results = response.json()
```

### cURL

```bash
# Health check (no auth)
curl http://localhost:8000/health

# Query (auth optional)
curl -X POST http://localhost:8000/query \
  -u admin:admin123 \
  -H "Content-Type: application/json" \
  -d '{"query": "Aspirin", "top_k": 5}'

# Ingest (auth required)
curl -X POST http://localhost:8000/ingest \
  -u admin:admin123 \
  -H "Content-Type: application/json" \
  -d @substance.json

# Batch ingest (auth required)
curl -X POST http://localhost:8000/ingest/batch \
  -u admin:admin123 \
  -H "Content-Type: application/json" \
  -d '{"substances": [...]}'

# Delete (auth required)
curl -X DELETE http://localhost:8000/substances/0103a288-6eb6-4ced-b13a-849cd7edf028 \
  -u admin:admin123

# Statistics (auth required)
curl -u admin:admin123 http://localhost:8000/statistics
```

### JavaScript (fetch)

```javascript
// Using btoa for Basic Auth
const credentials = btoa('admin:admin123');
const headers = {
  'Authorization': `Basic ${credentials}`,
  'Content-Type': 'application/json'
};

// Query
const response = await fetch('http://localhost:8000/query', {
  method: 'POST',
  headers,
  body: JSON.stringify({query: 'Aspirin', top_k: 5})
});
const results = await response.json();

// Ingest
const ingestResponse = await fetch('http://localhost:8000/ingest', {
  method: 'POST',
  headers,
  body: JSON.stringify({substance: substanceData})
});
```

### JavaScript (axios)

```javascript
import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000',
  auth: {
    username: 'admin',
    password: 'admin123'
  }
});

// Query
const { data } = await api.post('/query', {
  query: 'Aspirin',
  top_k: 5
});

console.log(data.results);
```

---

## Data Models

### VectorDocument

```typescript
interface VectorDocument {
  chunk_id: string;           // Unique chunk identifier
  document_id: string;        // Substance UUID
  section: string;            // Section name (e.g., "codes", "names")
  text: string;               // Chunk text content
  chunk_metadata: object;     // Additional metadata
  source_url?: string;        // Optional source URL
}
```

### QueryResult

```typescript
interface QueryResult {
  chunk_id: string;
  document_id: string;
  section: string;
  text: string;
  similarity_score: number;
  chunk_metadata: object;
}
```

### IngestRequest

```typescript
interface IngestRequest {
  substance: GSRSSubstance;   // GSRS Substance JSON
}
```

### IngestResponse

```typescript
interface IngestResponse {
  substance_uuid: string;
  chunks_created: number;
  element_paths: string[];
}
```

### QueryRequest

```typescript
interface QueryRequest {
  query: string;
  top_k?: number;
  filters?: object;
}
```

### QueryResponse

```typescript
interface QueryResponse {
  query: string;
  results: QueryResult[];
  total_results: number;
}
```

---

## Related Documentation

- [Quick Start](quickstart.md)
- [Configuration](configuration.md)
- [Authentication](authentication.md)
- [Guides](guides/README.md)
