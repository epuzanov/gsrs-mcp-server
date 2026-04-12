# Authentication Guide

The GSRS MCP Server supports **HTTP Basic Authentication** for API access.

## Quick Start

### Default Credentials

On first startup, the gateway uses default credentials configured via environment variables:

- **Username**: `admin` (or set via `API_USERNAME`)
- **Password**: `admin123` (or set via `API_PASSWORD`)

**⚠️ Change the default credentials in production!**

## HTTP Basic Authentication

### Using curl

```bash
# With -u flag (username:password)
curl -u admin:admin123 http://localhost:8000/health

# Ingest a substance
curl -X POST http://localhost:8000/ingest \
  -u admin:admin123 \
  -H "Content-Type: application/json" \
  -d @substance.json

# Query the gateway
curl -X POST http://localhost:8000/query \
  -u admin:admin123 \
  -H "Content-Type: application/json" \
  -d '{"query": "aspirin", "top_k": 5}'

# Delete a substance
curl -X DELETE http://localhost:8000/substances/0103a288-6eb6-4ced-b13a-849cd7edf028 \
  -u admin:admin123
```

### Using Authorization Header

```bash
# Generate Base64 encoded credentials
echo -n "admin:admin123" | base64
# Output: YWRtaW46YWRtaW4xMjM=

# Use in Authorization header
curl -X POST http://localhost:8000/query \
  -H "Authorization: Basic YWRtaW46YWRtaW4xMjM=" \
  -H "Content-Type: application/json" \
  -d '{"query": "aspirin", "top_k": 5}'
```

## Python Examples

### Using httpx

```python
import httpx

# Using auth tuple (recommended)
response = httpx.post(
    "http://localhost:8000/query",
    auth=("admin", "admin123"),
    json={"query": "aspirin", "top_k": 5}
)
print(response.json())

# Ingest substance
with open("substance.json") as f:
    response = httpx.post(
        "http://localhost:8000/ingest",
        auth=("admin", "admin123"),
        json={"substance": f.read()}
    )
print(response.json())
```

### Using requests

```python
import requests

# Using auth tuple
response = requests.post(
    "http://localhost:8000/query",
    auth=("admin", "admin123"),
    json={"query": "aspirin", "top_k": 5}
)
print(response.json())

# Get statistics
response = requests.get(
    "http://localhost:8000/statistics",
    auth=("admin", "admin123")
)
print(response.json())
```

### Using OpenAI SDK (for ChatGPT integration)

```python
from openai import OpenAI
import base64

# Create Basic Auth header
credentials = base64.b64encode(b"admin:admin123").decode()

# Configure OpenAI client to use local MCP Server
client = OpenAI(
    api_key="not-needed",  # API key not used with Basic Auth
    base_url="http://localhost:8000"
)

# Add custom headers for Basic Auth
# Note: You'll need to use httpx directly for Basic Auth with OpenAI SDK
import httpx

client = httpx.Client(
    base_url="http://localhost:8000",
    auth=("admin", "admin123")
)

response = client.post("/query", json={
    "query": "What is the CAS code for Aspirin?",
    "top_k": 5
})
print(response.json())
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_USERNAME` | `admin` | Username for Basic Authentication |
| `API_PASSWORD` | `admin123` | Password for Basic Authentication |

### Configure in .env

```bash
# Change default credentials
API_USERNAME=your-username
API_PASSWORD=your-secure-password
```

## Security Best Practices

### 1. Change Default Credentials

**Immediately after installation:**

```bash
# Edit .env file
cp .env.example .env

# Set strong credentials
API_USERNAME=your-unique-username
API_PASSWORD=$(openssl rand -base64 32)  # Generate random password
```

### 2. Use Strong Passwords

```bash
# Generate secure random password
openssl rand -base64 32

# Or use a password manager
```

### 3. Enable HTTPS in Production

Always use HTTPS to protect credentials in transit:

```bash
# With nginx reverse proxy
# Or use a service like ngrok for testing
ngrok http 8000
```

### 4. Restrict Network Access

```bash
# Bind to localhost only (development)
API_HOST=127.0.0.1

# Use firewall rules to restrict access
# Use Docker network isolation
```

### 5. Use Separate Credentials per Environment

```bash
# Development
API_USERNAME=dev_user
API_PASSWORD=dev_password

# Production
API_USERNAME=prod_user
API_PASSWORD=very_secure_production_password
```

## Protected Endpoints

The following endpoints require authentication:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ingest` | POST | Ingest a new substance |
| `/ingest/batch` | POST | Batch ingest multiple substances |
| `/substances/{uuid}` | DELETE | Delete a substance |
| `/statistics` | GET | Get database statistics |
| `/substance-classes` | GET | List available substance classes |
| `/models` | GET | Get available embedding models |

### Public Endpoints (No Auth Required)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/query` | POST | Query the gateway (optional auth) |

## Troubleshooting

### 401 Unauthorized

```json
{
  "detail": "Could not validate credentials"
}
```

**Solutions:**
- Check username and password are correct
- Ensure credentials are Base64 encoded correctly
- Verify no extra spaces in credentials

### Connection Refused

```
curl: (7) Failed to connect to localhost port 8000
```

**Solutions:**
- Check if the gateway is running: `docker-compose ps`
- Verify port mapping: `docker-compose logs rag-gateway`

### Invalid Base64 Encoding

```bash
# Correct encoding (no trailing newline)
echo -n "admin:admin123" | base64

# Wrong encoding (includes newline)
echo "admin:admin123" | base64
```

## Authentication Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Client    │────▶│  RAG API     │────▶│  Database   │
│             │     │  /query      │     │  (pgvector) │
└─────────────┘     └──────────────┘     └─────────────┘
       │
       │ Authorization: Basic base64(username:password)
       ▼
┌─────────────┐
│  Auth Layer │
│  (Basic)    │
└─────────────┘
```

## Related Documentation

- [API Reference](api-reference.md)
- [Configuration](configuration.md)
- [Quick Start](quickstart.md)
