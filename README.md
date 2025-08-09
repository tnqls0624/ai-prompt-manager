# FastMCP Prompt Enhancement Server

Production-ready MCP server using FastMCP, ChromaDB vector storage, and local LLM via Ollama for intelligent prompt enhancement.

## Architecture

```
Cursor/IDE → FastMCP Server (MCP + SSE)
                 ├─ MCP Tools
                 ├─ LangChain RAG Pipeline
                 │     ├─ Retriever → VectorService → ChromaDB
                 │     └─ LLM → Ollama
                 └─ Indexing/Watcher/Feedback Services
```

## LLM Usage

The server uses LLM in two modes:

### 1. **With LLM (Full Features)**

- **Model**: `r1-1776:latest` via Ollama
- **Embeddings**: `nomic-embed-text` (Nomic AI)
- **Capabilities**:
  - AI-powered prompt enhancement
  - Context-aware code generation
  - Intelligent summarization

### 2. **Without LLM (Fallback Mode)**

- Still functional with template-based enhancement
- Uses `StandardPromptFormatter` for structured improvements
- Vector search and context retrieval remain available

## Core Components

### 1. MCP Tools (15 available)

**LLM-Powered Tools:**

- `enhance_prompt` - AI-powered prompt improvement (uses LLM when available)
- `get_prompt_recommendations` - Context-aware recommendations
- `generate_test_skeleton` - Minimal failing test scaffolding (LLM-aware with fallback)

**Vector Search Tools (No LLM Required):**

- `store_conversation` - Persist user-AI interactions
- `search_similar_conversations` - Semantic search using embeddings
- `search_project_files` - Search indexed project files
- `get_project_context_info` - Project context retrieval

**Analytics Tools (No LLM Required):**

- `analyze_conversation_patterns` - Pattern analysis
- `analyze_prompt_patterns` - K-means clustering
- `extract_prompt_keywords` - TF-IDF keyword extraction
- `analyze_prompt_trends` - Temporal trend analysis

**Feedback Tools:**

- `submit_user_feedback` - Feedback loop
- `get_feedback_statistics` - Metrics and analytics
- `analyze_feedback_patterns` - Pattern recognition

**System Tools:**

- `get_fast_indexing_stats` - Performance metrics
- `get_server_status` - Health check

### 2. REST API Endpoints

**LLM-Required Endpoints:**

- `/api/v1/rag/enhance-prompt` - LangChain RAG enhancement with LLM
- `/api/v1/rag/generate-code` - Code generation (primary LLM usage)
- `/api/v1/rag/search-summarize` - Search and summarize with LLM

**LLM-Optional Endpoints:**

- `/api/v1/enhance-prompt-stream/{connection_id}` - Streaming enhancement (fallback available)
- `/api/v1/sse/{connection_id}` - Server-sent events
- `/api/v1/upload-batch` - Batch file upload
- `/api/v1/watcher/start` - File system monitoring
- `/api/v1/feedback` - Feedback submission
- `/api/v1/heartbeat` - Server health check
- `/api/v1/validate` - System/indexing/LLM health and readiness check
- `/api/v1/resource/snippet` - Return a file snippet by path and line range
- `/api/v1/rag/generate-edit` - Generate minimal JSON edits for code changes
- `/metrics` - Prometheus metrics (optional)
- `/dashboard` - Simple HTML dashboard (auto refresh)
- `/api/v1/index/warmup/{project_id}` - Build TF-IDF/BM25 caches for a project
- `/api/v1/audit/recent` - Recent audit items (JSONL)
- `/api/v1/audit/search` - Search audit logs by project/event/since
- `/dashboard` - Simple HTML dashboard (auto refresh)

### 3. Services

- **VectorService** - ChromaDB integration with Nomic embeddings
- **PromptEnhancementService** - Core prompt improvement (LLM with fallback)
- **FastIndexingService** - Parallel file indexing (100+ files concurrently)
- **LangChainRAGService** - RAG pipeline with LLM integration
- **FileWatcherService** - Real-time file monitoring
- **FeedbackService** - User feedback processing
- **AdvancedAnalyticsService** - ML-powered analytics (clustering, TF-IDF)

## Performance

Optimized for high throughput with actual benchmarks:

- **Concurrent Processing**: 100 simultaneous requests
- **File Processing**: 200 files in parallel
- **Embedding Batch**: 100 documents per batch
- **ChromaDB Batch**: 500 vectors per write
- **Connection Pool**: 100 persistent HTTP connections
- **LLM Requests**: Async with timeout handling
- **Hybrid Search**: Parallel semantic + TF-IDF with project-scoped cache
- **Reranking**: Tunable weights for semantic/keyword/recency/complexity
- **TF-IDF Index**: Cached per project with TTL to avoid recomputation
- **Persistent Index Cache**: Vectorizer + matrix persisted under `cache_dir`

## Quick Start

### 1. Using Docker (Recommended)

```bash
# Start all services (including Ollama)
docker-compose up -d

# Check if Ollama model is loaded
docker exec deepseek-r1-server ollama list

# If model not loaded, pull it
docker exec deepseek-r1-server ollama pull r1-1776

# View logs
docker-compose logs -f fastmcp-server
```

### 2. Direct Python

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure Ollama is running locally
ollama serve

# Pull the model
ollama pull r1-1776
ollama pull nomic-embed-text

# Run server
python mcp_server.py
```

## Configuration

Key environment variables in `docker-compose.yml`:

```yaml
environment:
  # Model Configuration
  - EMBEDDING_MODEL_TYPE=deepseek # Configuration name (uses Ollama backend)
  - DEEPSEEK_API_BASE=http://deepseek-r1:11434 # Ollama server endpoint
  - DEEPSEEK_EMBEDDING_MODEL=nomic-embed-text # Nomic AI embedding model
  - DEEPSEEK_LLM_MODEL=r1-1776:latest # LLM model via Ollama

  # Performance Settings
  - MAX_CONCURRENT_REQUESTS=100
  - EMBEDDING_BATCH_SIZE=100
  - CHROMA_BATCH_SIZE=500

  # Hybrid Search Weights (optional)
  - HYBRID_SEMANTIC_WEIGHT=0.7
  - HYBRID_KEYWORD_WEIGHT=0.3
  - RECENCY_WEIGHT=0.1
  - COMPLEXITY_WEIGHT=0.1
  - TFIDF_INDEX_TTL_SECONDS=300
  - CACHE_DIR=/data/cache

  # Warmup at start (optional)
  - WARMUP_ON_START=false
  - WARMUP_PROJECT_IDS=my-project,another-project

  # Audit log (optional)
  - AUDIT_LOG_ENABLED=false

  # Auth (disabled by default)
  - REQUIRE_API_KEY=false
  - API_KEY=
  - JWT_ENABLED=false
  - JWT_SECRET=
  - JWT_ALGORITHMS=HS256
  - PROJECT_QUOTA_PER_MINUTE=0

  # Warmup at start (optional)
  - WARMUP_ON_START=false
  - WARMUP_PROJECT_IDS=my-project,another-project

  # Audit log (optional)
  - AUDIT_LOG_ENABLED=false
```

**Note**: Variable names use "deepseek" prefix for historical reasons. Actual models:

- **Embeddings**: Nomic AI's `nomic-embed-text` (1.5GB, 768 dimensions)
- **LLM**: `r1-1776:latest` via Ollama (size varies)

## Project Upload

### High-performance batch upload:

```bash
python scripts/fast_upload.py /path/to/project --project-id my-project
```

Features:

- Parallel file reading (50 concurrent)
- Batch API calls (300 files per request)
- Automatic retry with exponential backoff
- Progress tracking

## MCP Integration

### Cursor Setup

Add to your MCP settings:

```json
{
  "mcpServers": {
    "prompt-enhancement": {
      "command": "python",
      "args": ["/path/to/mcp_server.py"],
      "env": {
        "EMBEDDING_MODEL_TYPE": "deepseek",
        "DEEPSEEK_EMBEDDING_MODEL": "nomic-embed-text",
        "DEEPSEEK_LLM_MODEL": "r1-1776:latest",
        "DEEPSEEK_API_BASE": "http://localhost:11434"
      }
    }
  }
}
```

### Usage Examples

#### With LLM:

```python
# Full AI-powered enhancement
result = await enhance_prompt(
    prompt="Build a React component",
    project_id="my-project",
    context_limit=5
)
# Returns: AI-generated improved prompt with context
```

#### Without LLM (Fallback):

```python
# Same call, but returns template-based enhancement
result = await enhance_prompt(
    prompt="Build a React component",
    project_id="my-project",
    context_limit=5
)
# Returns: Structured template with context, no AI generation
```

#### Generate Test Skeleton (TDD Red phase helper):

#### Streamed Code Generation (SSE):

```bash
curl -N -X POST http://localhost:8000/api/v1/rag/generate-code \
  -H 'content-type: application/json' \
  -d '{
        "prompt":"Implement user login API",
        "project_id":"my-project",
        "context_limit":5,
        "stream":true
      }'
```

#### Get File Snippet:

```bash
curl "http://localhost:8000/api/v1/resource/snippet?file_path=/host_projects/myproj/app.py&start_line=10&end_line=60"
```

#### Generate Minimal JSON Edits:

```bash
curl -X POST http://localhost:8000/api/v1/rag/generate-edit \
  -H 'content-type: application/json' \
  -d '{
        "instruction":"Rename function foo to fetch_user",
        "project_id":"my-project",
        "file_path":"/host_projects/myproj/app.py",
        "diff_context":"def foo(...):\n    pass\n"
      }'
```

```python
result = await generate_test_skeleton(
    feature="user can reset password via token",
    framework="pytest",           # or "jest", "unittest"
    project_id="my-project"
)
print(result["content"])  # test file content
```

## Technical Stack

- **FastMCP 2.9.0** - MCP protocol implementation
- **ChromaDB 0.4.22** - Vector database
- **LangChain 0.1.5** - RAG pipeline and LLM orchestration
- **Ollama** - Local LLM server
  - `r1-1776:latest` - Language model
  - `nomic-embed-text` - Embedding model
- **scikit-learn 1.3.0** - ML algorithms (clustering, TF-IDF)
- **SSE/WebSocket** - Real-time communication

## Resource Requirements

### Minimum (Without LLM):

- **Memory**: 2GB
- **CPU**: 2 cores
- **Storage**: 1GB + data

### Recommended (With LLM):

- **Memory**: 8GB (more for larger models)
- **CPU**: 4 cores
- **Storage**: SSD with 20GB+ for models
- **Docker**: 6GB memory allocation

## Development

### Running Tests

```bash
python -m pytest tests/ -v
```

### Adding New MCP Tools

```python
@mcp.tool()
async def your_new_tool(param: str) -> Dict[str, Any]:
    """Tool description"""
    # Can use self.llm if available
    if self.llm:
        result = await self.llm.arun(prompt)
    else:
        result = fallback_logic()
    return result
```

## Architecture Decisions

1. **FastMCP over raw MCP**: Better performance, built-in SSE support
2. **ChromaDB over alternatives**: Best local vector DB performance
3. **Ollama for LLM**: Local execution, privacy, no API costs
4. **Nomic embeddings**: Open-source, efficient, good quality
5. **Fallback mechanisms**: Service remains functional without LLM
6. **Parallel processing**: 5-10x performance gains

## Monitoring

```bash
# Check LLM availability
curl http://localhost:11434/api/tags

# Performance stats
curl http://localhost:8000/api/v1/heartbeat

# Error tracking
docker-compose logs fastmcp-server | grep ERROR

# ChromaDB health
curl http://localhost:8001/api/v1/heartbeat

# System validation (LLM/indexing/errors/perf)
curl "http://localhost:8000/api/v1/validate?project_id=my-project"

# Prometheus metrics (optional)
curl http://localhost:8000/metrics

# Simple Dashboard (auto refresh)
open http://localhost:8000/dashboard
```

## Known Limitations

- Maximum file size: 50MB per file
- ChromaDB collection limit: 1M vectors
- Concurrent connections: 100 (configurable)
- LLM context window: Model-dependent (typically 8K-32K tokens)
- LLM response time: 1-10 seconds depending on prompt complexity

## Troubleshooting

### LLM Not Working?

```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Verify model is loaded
ollama list

# Pull model if missing
ollama pull r1-1776
```

### Fallback to Template Mode

- Service automatically falls back if LLM is unavailable
- Check logs for "LLM 초기화 실패" messages
- Template-based enhancement still provides structured improvements

## Contributing

Pull requests welcome. Focus on:

- Performance improvements
- New MCP tool implementations
- Better LLM prompt engineering
- Enhanced fallback mechanisms
- Test coverage

---

Built for production. Works with or without LLM. Just works.
