-- game_api_service.lua
-- Defines the Lua game-related functions that can be called by the Python backend.

local GameApiService = {}
GameApiService.__index = GameApiService

-- --- Game State Simulation ---
-- This should ideally be accessed from a global game state manager
-- or passed into the functions if they are methods of a game object.
-- For this example, it's encapsulated within the service instance.
local _game_state = {
    player_position = {x = 0, y = 0},
    game_time = 0,
    is_paused = false
}

function GameApiService:new()
    -- Each instance of the service can have its own game_state
    -- If you want a singleton game_state, remove this and use a module-level table.
    local obj = setmetatable({
        game_state = {
            player_position = {x = 0, y = 0},
            game_time = 0,
            is_paused = false
        }
    }, self)
    print("[GameAPI-Service] New instance created. Initial player position: " .. obj.game_state.player_position.x .. "," .. obj.game_state.player_position.y)
    return obj
end

--- Moves the player within the game world.
-- @param direction string The direction to move ("north", "south", "east", "west").
-- @return boolean true if successful, false otherwise.
-- @return string|nil An error message if failed.
function GameApiService:move_player(direction)
    if type(direction) ~= "string" then
        error("Argument 'direction' must be a string.")
    end

    print(string.format("[GameAPI-Service] Attempting to move player: %s", direction))
    local moved = false
    if direction == "north" then
        self.game_state.player_position.y = self.game_state.player_position.y + 1
        moved = true
    elseif direction == "south" then
        self.game_state.player_position.y = self.game_state.player_position.y - 1
        moved = true
    elseif direction == "east" then
        self.game_state.player_position.x = self.game_state.player_position.x + 1
        moved = true
    elseif direction == "west" then
        self.game_state.player_position.x = self.game_state.player_position.x - 1
        moved = true
    else
        error("Invalid direction specified. Use 'north', 'south', 'east', or 'west'.")
    end

    if moved then
        print(string.format("[GameAPI-Service] Player new position: x=%d, y=%d", self.game_state.player_position.x, self.game_state.player_position.y))
        return true, "Player moved successfully."
    end
    -- This path should ideally not be reached if direction is valid
    return false, "Failed to move player for unknown reason."
end

--- Gets the current player position.
-- @return table A table with x and y coordinates.
function GameApiService:get_player_position()
    print(string.format("[GameAPI-Service] Request to get player position. Current: x=%d, y=%d", self.game_state.player_position.x, self.game_state.player_position.y))
    return self.game_state.player_position
end

--- Sets the game's pause state.
-- @param is_paused boolean true to pause, false to unpause.
-- @return boolean The new pause state.
function GameApiService:set_game_pause(is_paused)
    if type(is_paused) ~= "boolean" then
        error("Invalid argument: is_paused must be a boolean.")
    end
    self.game_state.is_paused = is_paused
    print(string.format("[GameAPI-Service] Game pause state set to: %s", tostring(is_paused)))
    return self.game_state.is_paused
end

--- Gets the current simulated game time.
-- @return number The current game time in seconds.
function GameApiService:get_game_time()
    print(string.format("[GameAPI-Service] Request to get game time. Current: %.2f seconds", self.game_state.game_time))
    return self.game_state.game_time
end

--- Function to update the game state (can be called by a main loop or RPC)
-- @param dt number Delta time for update.
function GameApiService:update_game_state(dt)
    if type(dt) ~= "number" then
        error("Argument 'dt' must be a number.")
    end
    if not self.game_state.is_paused then
        self.game_state.game_time = self.game_state.game_time + dt
        print(string.format("[GameAPI-Service] Game state updated. New time: %.2f", self.game_state.game_time))
    else
        print("[GameAPI-Service] Game is paused, not updating time.")
    end
    return true
end

--- Function to get raw game_state (for testing or internal use).
-- @return table The internal game state table.
function GameApiService:get_game_state()
    print("[GameAPI-Service] Getting raw game state.")
    return self.game_state
end

--- Get exposed methods for introspection.
-- @return table A sorted list of method names available on this service.
function GameApiService:get_exposed_methods()
    local methods = {}
    for k, v in pairs(self) do
        if type(v) == "function" and k ~= "new" and k ~= "get_exposed_methods" then
            table.insert(methods, k)
        end
    end
    table.sort(methods)
    return methods
end

return GameApiService