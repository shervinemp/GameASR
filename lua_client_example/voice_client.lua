-- main.lua
package.path = package.path .. ';../voice_control/bridge/clients/lua/?.lua'
local RpcToolServer = require("rpc_tool_server")
local json = require("json")

-- --- ZeroMQ Endpoint Configuration (via Environment Variables) ---
-- These environment variables will be passed directly to the constructor.
-- If not set, the 'ToolServer:new' method will apply its own defaults.
--
-- TCP Example:
--   Linux/macOS: export LUA_CLIENT_PROTOCOL="tcp" LUA_CLIENT_ENDPOINT="127.0.0.1:5555"
--   Windows CMD: set LUA_CLIENT_PROTOCOL=tcp & set LUA_CLIENT_ENDPOINT=127.0.0.1:5555
--
-- IPC Example (Unix-like):
--   Linux/macOS: export LUA_CLIENT_PROTOCOL="ipc" LUA_CLIENT_ENDPOINT="/tmp/voice_control_service.ipc"
--
-- IPC Example (Windows Named Pipe):
--   Windows CMD: set LUA_CLIENT_PROTOCOL=ipc & set LUA_CLIENT_ENDPOINT=\\.\pipe\voice_control_service
--
local client_protocol = os.getenv("LUA_CLIENT_PROTOCOL")
local client_endpoint = os.getenv("LUA_CLIENT_ENDPOINT")

if not client_protocol then
    print("[Lua-Client] LUA_CLIENT_PROTOCOL env var not set. ToolServer:new will use its default ('tcp').")
end
if not client_endpoint then
    print("[Lua-Client] LUA_CLIENT_ENDPOINT env var not set. ToolServer:new will use its default ('127.0.0.1:5555' for TCP).")
end

print("--- Starting Lua ZeroMQ Client Example ---")

-- Pass arguments, which will be 'nil' if environment variables were not set.
-- The RpcToolServer:new constructor handles these 'nil' values by applying its defaults.
local client = RpcToolServer:new(client_protocol, client_endpoint)

local connected, connect_err = client:connect()
if not connected then
    print("FATAL ERROR: Failed to connect client: " .. connect_err)
    os.exit(1)
end

-- --- Test Cases ---

-- 1. set_contexts
local set_contexts_result, set_contexts_err = client:set_contexts({"music_control", "home_automation"})
if set_contexts_err then
    print("Set contexts failed.")
else
    print("Set contexts result:", json.encode(set_contexts_result))
end
print("------------------------------------------")

-- 2. query_ai for time
local query_result_time, query_err_time = client:query_ai("What time is it in Ottawa?")
if query_err_time then
    print("Query for time failed.")
else
    print("Query for time result:", json.encode(query_result_time))
end
print("------------------------------------------")

-- 3. query_ai for lights
local query_result_lights, query_err_lights = client:query_ai("Turn off the living room lights.")
if query_err_lights then
    print("Query for lights failed.")
else
    print("Query for lights result:", json.encode(query_result_lights))
end
print("------------------------------------------")

-- 4. Non-existent method (should return RPC error)
local bad_method_result, bad_method_err = client:query_ai("this_method_does_not_exist_on_server")
if bad_method_err then
    print("Expected RPC error for non-existent method:", bad_method_err)
else
    print("Unexpected success for non-existent method:", json.encode(bad_method_result))
end
print("------------------------------------------")

-- 5. Server-side error (if Python server can simulate one)
local server_error_result, server_error_err = client:query_ai("trigger_server_side_error")
if server_error_err then
    print("Expected server-side error:", server_error_err)
else
    print("Unexpected success for server-side error:", json.encode(server_error_result))
end
print("------------------------------------------")

print("--- Disconnecting Lua ZeroMQ Client ---")
client:disconnect()
print("--- Lua ZeroMQ Client Example Finished ---")