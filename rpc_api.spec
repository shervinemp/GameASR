{
  "methods": [
    {
      "type": "function",
      "function": {
        "name": "move_player",
        "description": "Moves the player within the game world. Returns a status message.",
        "parameters": {
          "type": "object",
          "properties": {
            "direction": {
              "type": "string",
              "description": "The direction to move ('north', 'south', 'east', 'west')."
            }
          },
          "required": ["direction"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "get_player_position",
        "description": "Gets the current player position. Returns a string describing the position.",
        "parameters": {
          "type": "object",
          "properties": {},
          "required": []
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "set_game_pause",
        "description": "Sets the game's pause state. Returns a status message.",
        "parameters": {
          "type": "object",
          "properties": {
            "is_paused": {
              "type": "boolean",
              "description": "true to pause, false to unpause."
            }
          },
          "required": ["is_paused"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "get_game_time",
        "description": "Gets the current simulated game time. Returns a string with the time.",
        "parameters": {
          "type": "object",
          "properties": {},
          "required": []
        }
      }
    }
  ]
}