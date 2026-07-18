-- main.lua
local LLMClient = require("llm_client")
local json = require("json")

-- --- ZeroMQ Endpoint Configuration (via Environment Variables) ---
local client_protocol = os.getenv("LUA_CLIENT_PROTOCOL")
local client_endpoint = os.getenv("LUA_CLIENT_ENDPOINT")
local auth_token = os.getenv("RPC_AUTH_TOKEN")

print("--- Starting Lua LLM Client Example ---")

local client = LLMClient:new(client_protocol, client_endpoint, auth_token)

local connected, connect_err = client:connect()
if not connected then
    print("FATAL ERROR: Failed to connect client: " .. connect_err)
    os.exit(1)
end

local query_result, query_error = client:query_sync("Move alpha to 10, 20.")
if query_error then
    print("Query failed:", query_error)
else
    print("Query result:", json.encode(query_result))
end
print("------------------------------------------")

print("--- Disconnecting Lua ZeroMQ Client ---")
client:disconnect()
print("--- Lua ZeroMQ Client Example Finished ---")
