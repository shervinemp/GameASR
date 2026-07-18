-- Example functions exposed by a Lua game to the voice-control tool client.

local rpc_api = {}

--[[
    Moves a specified unit to a target location.
    @param name (string): The name of the unit to move.
    @param x (number): The target x-coordinate.
    @param y (number): The target y-coordinate.
]]
function rpc_api.move_unit(name, x, y)
    if GameState.current.squad and GameState.current.squad[name] then
        local unit = GameState.current.squad[name]
        unit.target_x = x
        unit.target_y = y
        return {status = "success"}
    end
    return {status = "error", message = "Unit not found: " .. tostring(name)}
end

--[[
    Commands a unit to perform an action.
    @param name (string): The name of the unit to command.
    @param action (string): The action to perform.
    @param target_x (number): The action target's x-coordinate.
    @param target_y (number): The action target's y-coordinate.
]]
function rpc_api.command_unit(name, action, target_x, target_y)
    if GameState.current.squad and GameState.current.squad[name] then
        local unit = GameState.current.squad[name]
        if action == "shoot" then
            unit:shoot_at(target_x, target_y)
            return {status = "success"}
        end
        return {status = "error", message = "Unknown action: " .. tostring(action)}
    end
    return {status = "error", message = "Unit not found: " .. tostring(name)}
end

--[[
    Sets the microphone listening status for a UI indicator.
    @param active (boolean): Whether the microphone is active.
]]
function rpc_api.set_mic_status(active)
    _G.mic_active = active
    return {status = "success"}
end

--[[
    Retrieves the status and position of all units in the squad.
]]
function rpc_api.get_squad_status()
    local status = {}
    if GameState.current.squad then
        for name, unit in pairs(GameState.current.squad) do
            status[name] = {x = unit.x, y = unit.y}
        end
        return {status = "success", data = status}
    end
    return {status = "error", message = "Squad not initialized."}
end

return rpc_api
