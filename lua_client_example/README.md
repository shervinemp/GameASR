# Lua Bridge Client Example

Example game integration using the voice‑control pipeline's Lua bridge client.

## Structure

```
lua_client_example/
├── rpc_api.lua          # RPC functions callable by the voice pipeline
├── voice_client.lua     # Example client connecting to the LLM bridge server
└── game/                # Simple game simulation (units, squads, physics, etc.)
    ├── init.lua
    └── lua/
        ├── abstractions/   # io.lua, input.lua, physics.lua, rendering.lua, etc.
        └── game_states.lua
```

## How it works

1. The voice‑control bridge server exposes a ZMQ endpoint.
2. `voice_client.lua` connects to the bridge and sends voice‑parsed commands.
3. The bridge calls `rpc_api.lua` functions (`move_unit`, `command_unit`, `get_squad_status`) on the game.
4. The game simulation in `game/` responds and returns results back through the bridge.

## Running

Set the bridge endpoint via environment variables:

```bash
export LUA_CLIENT_PROTOCOL="tcp"
export LUA_CLIENT_ENDPOINT="localhost:5555"
lua voice_client.lua
```

The bridge server is started by the Python pipeline (`uv run python -m voice_control ...`).
