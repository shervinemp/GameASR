-- File: lua_client_example/rpc_api.lua
-- Defines the functions that the Python voice service can call.

local rpc_api = {}

--[[
    Moves a specified unit to a target location.
    @param name (string): The name of the unit to move (e.g., "player", "alpha").
    @param x (number): The target x-coordinate.
    @param y (number): The target y-coordinate.
]]
function rpc_api.move_unit(name, x, y)
    if GameState.current.squad and GameState.current.squad[name] then
        local unit = GameState.current.squad[name]
        unit.target_x = x
        unit.target_y = y
        return {status = "success", message = name .. " moving to " .. x .. "," .. y}
    end
    return {status = "error", message = "Unit not found: " .. tostring(name)}
end

--[[
    Commands a unit to perform an action, like shooting.
    @param name (string): The name of the unit to command.
    @param action (string): The action to perform (e.g., "shoot").
    @param target_x (number): The x-coordinate for the action's target.
    @param target_y (number): The y-coordinate for the action's target.
]]
function rpc_api.command_unit(name, action, target_x, target_y)
    if GameState.current.squad and GameState.current.squad[name] then
        local unit = GameState.current.squad[name]
        if action == "shoot" then
            unit:shoot_at(target_x, target_y)
            return {status = "success", message = name .. " is shooting at " .. target_x .. "," .. target_y}
        end
        return {status = "error", message = "Unknown action: " .. tostring(action)}
    end
    return {status = "error", message = "Unit not found: " .. tostring(name)}
end

--[[
    Retrieves the status and position of all units in the squad.
    @return (table): A table containing information for each unit.
]]
function rpc_api.get_squad_status()
    local status = {}
    if GameState.current.squad then
        for name, unit in pairs(GameState.current.squad) do
            status[name] = {
                x = unit.x,
                y = unit.y,
                target_x = unit.target_x,
                target_y = unit.target_y
            }
        end
        return {status = "success", data = status}
    end
    return {status = "error", message = "Squad not initialized."}
end

return rpc_api
