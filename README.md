# GameASR — Voice-Controlled Game Agent

A modular voice control pipeline that lets you control games and applications using natural speech. Features ASR (speech-to-text), LLM (intent parsing), TTS (voice feedback), and RAG over a knowledge graph — all with push-to-talk and game-state integration.

## Architecture

```
Game (Lua) ←→ Bridge Server (ZMQ) ←→ Pipeline ←→ LLM
                                       ↕
                              ASR ←→ TTS ←→ RAG
                                            ↕
                                        Neo4j KG
```

| Component | Role |
|-----------|------|
| **ASR** | Speech-to-text (ParakeetV2, Whisper, etc.) |
| **LLM** | Intent parsing + response generation (Ollama, OpenAI, Gemini, GGUF) |
| **TTS** | Text-to-speech feedback (Kokoro, etc.) |
| **RAG** | S-Path-RAG over a Neo4j knowledge graph |
| **Bridge** | ZMQ/TCP/IPC bridge to game engine (Lua, C++, C#, JS, Python) |

## Quick Start

```bash
# Install
uv sync

# Configure
cp voice_control/config.defaults.yaml config.yaml
# Set your LLM provider, API keys, Neo4j credentials, etc.

# Run (with web-search RAG):
uv run python -m voice_control api_spec.json

# Run (pipeline with push-to-talk):
uv run python -m voice_control.pipeline
```

## Configuration

Create `config.yaml` in the project root to override defaults from `voice_control/config.defaults.yaml`:

```yaml
llm:
  provider: "ollama"       # or openai, gemini, Gemma4E2B
  providers:
    ollama:
      model: "qwen3:latest"

database:
  neo4j:
    uri: "bolt://localhost:7687"
    user: "neo4j"
    password: "your_password"   # or set NEO4J_PASSWORD env var
```

## RAG Pipeline

The project implements **S-Path-RAG** (Semantic Shortest-Path RAG):

- **Knowledge Graph**: Entities stored in Neo4j with vector + fulltext indexes
- **Strategies**: `NeighborhoodStrategy` (N-hop expansion) and `ShortestPathStrategy` (multi-anchor path discovery)
- **Iterative Socratic Loop**: Retrieves → critiques → refines query → re-retrieves until confident
- **Fallback**: Web search via DuckDuckGo when the graph yields no results

```bash
# Import CoDEx dataset into Neo4j:
uv run python -m voice_control.rag.data
```

## Bridge Clients

Pre-built clients for integrating with game engines:

| Language | File |
|----------|------|
| Lua | `lua_client_example/voice_client.lua` |
| C++ | `voice_control/bridge/clients/cpp/` |
| C# | `voice_control/bridge/clients/cs/` |
| JS | `voice_control/bridge/clients/js/` |
| Python | `voice_control/bridge/clients/python/` |
| GDScript | `voice_control/bridge/clients/gdscript/` |

## Testing

```bash
uv run python -m unittest discover -s tests -v
```

## License

MIT
