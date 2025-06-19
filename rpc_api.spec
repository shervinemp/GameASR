{
  "api_name": "LuaGameAPI",
  "description": "API exposed by the Lua game for the Python backend to call.",
  "methods": [
    {
      "name": "move_player",
      "description": "Moves the player within the game world. Returns a status message.",
      "params": [
        {"name": "direction", "type": "string", "description": "The direction to move ('north', 'south', 'east', 'west')."}
      ],
      "returns": {"type": "string"}
    },
    {
      "name": "get_player_position",
      "description": "Gets the current player position. Returns a string describing the position.",
      "params": [],
      "returns": {"type": "string"}
    },
    {
      "name": "set_game_pause",
      "description": "Sets the game's pause state. Returns a status message.",
      "params": [
        {"name": "is_paused", "type": "boolean", "description": "true to pause, false to unpause."}
      ],
      "returns": {"type": "string"}
    },
    {
      "name": "get_game_time",
      "description": "Gets the current simulated game time. Returns a string with the time.",
      "params": [],
      "returns": {"type": "string"}
    }
  ]
}