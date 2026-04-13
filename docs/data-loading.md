# Data Loading

The recommended loading path is the bundled loader script:

```bash
python scripts/load_data.py --uuids 0103a288-6eb6-4ced-b13a-849cd7edf028
```

You can also ingest through MCP with `gsrs_ingest`.

Readiness note:

- `/readyz` can be healthy even when the database is empty
- `gsrs_statistics` and `/health` will show `0` chunks and `0` substances until data is loaded
