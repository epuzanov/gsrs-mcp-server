# Authentication

## MCP HTTP Transport

When `API_USERNAME` and `API_PASSWORD` are set, the server enables bearer-token verification for the `/mcp` endpoint.

Use:

```http
Authorization: Bearer <API_PASSWORD>
```

Example:

```bash
curl http://localhost:8000/mcp \
  -H "Authorization: Bearer change-me"
```

## Health Endpoints

These endpoints do not require auth:

- `/livez`
- `/readyz`
- `/health`

## stdio Transport

No HTTP auth is used for `stdio` because the client launches the server locally.

## Recommendations

- Change `API_PASSWORD` immediately outside local development
- Prefer `stdio` for single-user local clients
- Prefer a reverse proxy and HTTPS for shared HTTP deployments
