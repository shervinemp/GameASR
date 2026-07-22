# GameASR — Voice-Controlled Game Agent

A modular voice control pipeline with graph-based RAG. ASR captures speech,
LLM parses intent, TTS responds, and a knowledge graph enriches answers with
structured facts via S-Path-RAG.

## Architecture

```mermaid
flowchart LR
    subgraph Game["Game Engine (Lua)"]
        Client["Voice Client"]
    end

    subgraph Pipeline["Voice Pipeline"]
        ASR["ASR (STT)"] -->|transcript| LLM["LLM (intent)"]
        LLM -->|response| TTS["TTS (voice)"]
    end

    subgraph RAG["RAG (S-Path)"]
        SGR["SmartGraph<br/>Retriever"] --> GS["GraphSearch"]
        GS --> RR["Reranker"]
        RR --> EV["Ranked Evidence"]
        RR -.->|direct RAG API| CP["Composer"]
    end

    KG[("Neo4j KG<br/>(vector + fulltext)")]

    Client <==>|ZMQ| Pipeline
    LLM -.->|retrieve tool| SGR
    EV -.->|untrusted evidence| LLM
    CP -.->|composed answer| LLM
    SGR <==> KG
```

## Features

- **ASR** — Speech-to-text (ParakeetV2, configurable) with push-to-talk
- **LLM** — Local GGUF models or LiteLLM remote backends (Ollama, OpenAI, Gemini)
- **TTS** — Voice feedback (Kokoro) with interrupt on new input
- **RAG** — S-Path-RAG over Neo4j with entity linking, anchor dedup, adaptive expansion, and Socratic correction loop
- **Bridge** — ZMQ/TCP/IPC bridge to game engines (Lua, C++, C#, JS, GDScript, Python)
- **Push-to-talk** — Configurable hotkey binding; speaking cuts off current TTS
- **Reviewable learning** — Optionally extracts new triplets into a local review queue before graph ingestion

## Prerequisites

- Python 3.11-3.13
- [uv](https://docs.astral.sh/uv/) (package manager)
- Neo4j 2025+ — for knowledge graph RAG
- [eSpeak NG](https://github.com/espeak-ng/espeak-ng) — for TTS phonemization (install to default path `C:\Program Files\eSpeak NG`)
- (Optional) Ollama, OpenAI key, or Gemini key for LLM backend

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Configure
cp config.example.yaml config.yaml
cp .env.example .env

# 3. Edit .env with your secrets
#    NEO4J_PASSWORD is required; API keys if using remote LLM backends

# 4. Import knowledge graph data (optional)
uv run python -m voice_control.rag.data

# 5. Run (bridge server with web-search RAG):
uv run python -m voice_control api_spec.json

# Or run (interactive pipeline):
uv run python -m voice_control.pipeline
```

## Configuration

**Which file does what:**

| File | Purpose |
|------|---------|
| `voice_control/config.defaults.yaml` | All defaults — ships with the project |
| `config.yaml` | User overrides — copy from `config.example.yaml` |
| `.env` | Secrets (gitignored) — copy from `.env.example` |
| `voice_control/common/model_manifest.yaml` | Download provenance for GGUF/TTS models |
| `voice_control/config.example.yaml` | Minimal config template |
| `.env.example` | Env var template |

### LLM backend selection

```yaml
llm:
  backend: "local"       # "local" (GGUF) or "litellm" (remote APIs)
  model: "Gemma4_12B"    # key into llm.local, or litellm sub-provider
  max_tool_iterations: 1

  # Local GGUF models — each key matches model_manifest.yaml
  local:
    Gemma4_12B:
      n_ctx: 8192
      decoder: "legacy_xml"
      type_k: "q4_0"
      type_v: "q4_0"
    # Local file (no manifest entry needed):
    # my_model:
    #   model_path: "model_files/llm/my_model.gguf"
    #   n_ctx: 4096

  # LiteLLM remote backends — used when backend is "litellm"
  litellm:
    provider: "ollama"
    model: "qwen3:latest"
    api_base: "http://localhost:11434"
```

### Adding a new GGUF model

1. Add download info to `model_manifest.yaml` (HuggingFace repo, file, sha256)
2. Add runtime settings to `llm.local` in `config.yaml` (n_ctx, decoder, etc.)

### Adding a local model file

Just add it to `llm.local` with a `model_path` — no manifest entry needed:

```yaml
llm:
  local:
    my_model:
      model_path: "/absolute/path/to/model.gguf"
      n_ctx: 4096
      decoder: "general"
```

### Role assignments (used by RAG)

```yaml
models:
  default: "llama3"
  extraction_heavy: "gemini-1.5-pro"
  embedding: "avsolatorio/GIST-small-Embedding-v0"
```

### Environment variables

Secrets are declared in `config.defaults.yaml` via `env:` blocks and resolved
at config load time. Set them in `.env`:

```bash
NEO4J_PASSWORD="password"
OPENAI_API_KEY="sk-..."
```

RPC binds to `127.0.0.1` by default. A non-loopback `--host` is rejected unless
`RPC_AUTH_TOKEN` is set to at least 32 characters. Put any remotely accessible
TCP bridge behind an encrypted transport such as a VPN or TLS tunnel.

## RAG Pipeline

The project implements **S-Path-RAG** (Semantic Shortest-Path Retrieval-Augmented Generation):

### Pipeline flow

1. **Entity linking** — Bounded query n-grams are matched through the indexed `normalized_label` property before NER or LLM extraction. Lowercase ASR transcripts still resolve known entities without a model call.
2. **Dual retrieval** — Unresolved entities use vector search while keyword search uses the fulltext index. Neighborhood vector and keyword searches run in parallel.
3. **Strategy execution**:
   - **NeighborhoodStrategy**: N-hop semantic expansion around matched entities. Auto-retries with `n_hops=2` when `n_hops=1` returns < 3 results.
   - **ShortestPathStrategy**: Pairs candidates only across distinct query entities, caps pair count, and uses Neo4j's native bounded `SHORTEST k` selector in one parameterized query.
4. **Source routing** — The game graph is queried first. Web search runs only when the graph returns no evidence and must finish inside the shared retrieval deadline.
5. **Reranking** — Candidates are bounded and deduplicated before the cross-encoder; identical reranking work and final contexts are cached under synchronized LRU caches.
6. **Evidence delivery** — The conversation's `retrieve` tool returns ranked `[graph]` or `[web]` evidence. The outer assistant performs the only user-facing synthesis, avoiding answer-to-answer generation.
7. **Direct composition** — Direct `rag(query)` calls remain supported. Internal prompts are stateless, short context bypasses lossy summarization, and a draft that passes critique is returned without regeneration.
8. **Reviewable learning** — Disabled by default. Approved learned nodes receive normalized labels and embeddings; cache entries are invalidated after graph writes.

### Optimizations

| Technique | Impact |
|-----------|--------|
| Indexed normalized-label linking | Catches known entities before NER, LLM extraction, or embedding |
| Parallel vector + keyword search | Both searches run concurrently |
| Cross-entity anchor pairing | Avoids paths between alternative matches for the same query entity |
| Native bounded shortest paths | Avoids enumerating and sorting every path up to three hops |
| Dedupe before reranker | Avoids scoring duplicate candidates while preserving coverage |
| Adaptive expansion (n_hops 1→2) | Auto-retries broader search when initial results are thin |
| Stateless one-shot internal prompts | Prevents hidden RAG history growth and cross-request contamination |
| Budget-aware summarization | Short evidence reaches generation unchanged and without an extra call |
| Graph-first web fallback | Removes network latency and untrusted web noise for graph-covered queries |
| Deadline and synchronized LRU caches | Bounds tail latency and skips repeated retrieval/reranking work |

After upgrading an existing graph, rerun `uv run python -m voice_control.rag.data`
to add the normalized-label index and rebuild entity embeddings from labels plus
descriptions. Runtime lookup retains a slower legacy fallback until migration.

### Key modules

| Module | Purpose |
|--------|---------|
| `rag/retrieval.py` | Reranker, graph strategies, SmartGraphRetriever, WebRetriever with fallback chain |
| `rag/knowledge.py` | Neo4j driver (vector search, keyword search, batch SPath, expansion, exact-label lookup) |
| `rag/model.py` | BaseRAG, SimpleRAG, SPathRAG orchestrators with active learning |
| `rag/generation.py` | Composer with Socratic correction, SLM-optimized prompts |
| `rag/triplet.py` | LLM-based knowledge triplet extraction |
| `rag/data.py` | CoDEx dataset import, entity/relationship import with source tracking |

### Knowledge Graph

Imported nodes and relationships carry provenance metadata. Approved learned
triplets can use `source: 'extraction'` and `created_at` for temporal queries:

```cypher
// All extracted (learned) knowledge
MATCH (n:Entity {source: 'extraction'})
// Relationships added after a specific time
MATCH ()-[r {source: 'extraction'}]->()
```

Import a CoDEx-formatted dataset:

```bash
uv run python -m voice_control.rag.data
```

## Bridge Clients

| Language | Path |
|----------|------|
| Lua | `lua_client_example/voice_client.lua` |
| C++ | `voice_control/bridge/clients/cpp/` |
| C# | `voice_control/bridge/clients/cs/` |
| JavaScript | `voice_control/bridge/clients/js/` |
| Python | `voice_control/bridge/clients/python/` |
| GDScript | `voice_control/bridge/clients/gdscript/` |

The bridge uses ZeroMQ (TCP or IPC). Game clients connect to the pipeline's
RPC server and expose functions via `rpc_api.lua`. Public/wildcard binds
require authentication; loopback is the secure default.

Bundled GGUF and TTS assets are downloaded only from allowlisted HTTPS
origins, at pinned revisions where applicable, and verified against committed
SHA-256 digests before loading.

### LLM provider routing

`litellm` backend routes through the in-process LiteLLM SDK and shares the same
streaming, tool-call, timeout, and error-handling path for OpenAI, Gemini, and
Ollama. No LiteLLM proxy process is started by this project.

Provider requests have a bounded timeout and automatic retries are disabled for
streaming calls, avoiding duplicate tool effects after a partial response. API
keys are resolved from environment variables and the project does not enable
LiteLLM debug logging, callbacks, or telemetry hooks. The bundled model-cost
map is used offline by default; set `LITELLM_LOCAL_MODEL_COST_MAP=False` before
startup to opt into LiteLLM's online cost-map refresh.

The `local` backend keeps using `llama-cpp-python` directly so local GGUF
model state and custom stream decoders remain available. Custom remote
`api_base` URLs must use HTTPS; plaintext HTTP is accepted only for loopback
services such as local Ollama.

## Push-to-Talk

- **Default hotkey**: <kbd>Right Ctrl</kbd>+<kbd>Right Shift</kbd>
- **Press and hold** to speak, **release** when done
- **Speaking while TTS is playing** interrupts the current output immediately
- **Press-to-reset**: <kbd>Left Ctrl</kbd>+<kbd>Right Ctrl</kbd> clears conversation history

## Project Structure

```
voice_control/
├── asr/                  # Speech-to-text providers
├── tts/                  # Text-to-speech (Kokoro)
├── llm/                  # Language model layer (session, conversation, tools)
├── rag/                  # Retrieval-augmented generation (S-Path-RAG)
├── bridge/               # Game engine bridge (ZMQ clients, RPC server)
├── pipeline.py           # Main orchestration pipeline
├── config.defaults.yaml  # Default configuration
└── __main__.py           # CLI entry point
```

## Testing

```bash
pytest tests/
```

## Contributing

PRs welcome. The codebase is under active development — open an issue first for
significant changes. No formal style guide yet; match the surrounding code.

## License

MIT
