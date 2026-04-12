# Contributing to GSRS MCP Server

Thank you for your interest in contributing to GSRS MCP Server! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what's best for the community

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- Clear title and description
- Steps to reproduce the issue
- Expected behavior vs actual behavior
- Python version and OS
- Any relevant logs or error messages

**Example:**
```markdown
**Bug**: ChromaDB connection fails on Windows

**Steps to Reproduce:**
1. Install with `pip install -r requirements.txt`
2. Set DATABASE_URL=chroma://./chroma_data
3. Run `uvicorn app.main:app`

**Expected:** Server starts successfully
**Actual:** Connection error

**Environment:**
- Python: 3.11
- OS: Windows 11
```

### Suggesting Features

Feature suggestions are welcome! Please provide:

- Use case and motivation
- Proposed solution
- Alternatives considered
- Additional context

### Pull Requests

1. Fork the repository
2. Create a branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/ -v`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/gsrs-rag-gateway.git
cd gsrs-rag-gateway

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies with dev extras
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linting
ruff check app/ tests/
mypy app/
```

### Code Style

- Follow PEP 8
- Use type hints where possible
- Write docstrings for public functions and classes
- Keep functions focused and small
- Write tests for new features

**Example:**
```python
def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """Generate embeddings for multiple texts.

    Args:
        texts: List of text strings to embed
        batch_size: Number of texts to process in each batch

    Returns:
        List of embedding vectors
    """
    if not texts:
        return []
    # ... implementation
```

### Testing

- Write tests for new features
- Ensure all tests pass (`pytest tests/ -v`)
- Maintain or improve code coverage
- Test with different backends (ChromaDB, pgvector)

### Documentation

- Update README.md if needed
- Add docstrings to new functions/classes
- Update API documentation for new endpoints
- Include examples in guides when appropriate

## Project Structure

```
gsrs-rag-gateway/
├── app/
│   ├── services/          # Core services (chunking, embeddings, db)
│   ├── db/                # Database backends
│   ├── models/            # SQLAlchemy models
│   └── main.py            # FastAPI application
├── tests/                 # Unit tests
├── docs/                  # Documentation
├── scripts/               # Utility scripts
└── examples/              # Integration examples
```

## Architecture

The gateway consists of three main services:

1. **ChunkerService** - Chunks GSRS Substance JSON into VectorDocuments
2. **EmbeddingService** - Generates embeddings via OpenAI/Ollama APIs
3. **VectorDatabaseService** - Manages vector database operations

## Release Process

Releases follow semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

## Questions?

Feel free to open an issue for any questions or discussions.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
