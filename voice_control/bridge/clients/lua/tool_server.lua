-- File: voice_control/bridge/clients/lua/tool_server.lua
-- This file implements the RPC server that listens for commands from the Python application.

local zmq = require("lzmq")
local json = require("json")

local ToolServer = {}
ToolServer.__index = ToolServer

function ToolServer:new(api, endpoint, auth_token)
    local obj = setmetatable({
        api = api,
        endpoint = endpoint or "tcp://127.0.0.1:8080",
        auth_token = auth_token or nil,
        context = nil,
        socket = nil,
        running = false,
        thread = nil,
    }, self)
    return obj
end

function ToolServer:_handle_request(request_str)
    local response = { jsonrpc = "2.0", id = nil }
    local request

    -- Safely decode JSON
    local ok, decoded = pcall(json.decode, request_str)
    if not ok then
        response.error = { code = -32700, message = "Parse error" }
        return json.encode(response)
    end
    request = decoded
    response.id = request.id

    -- Validate request
    if request.jsonrpc ~= "2.0" or not request.method then
        response.error = { code = -32600, message = "Invalid Request" }
        return json.encode(response)
    end

    -- Authentication
    if self.auth_token and request.auth_token ~= self.auth_token then
        response.error = { code = -32001, message = "Authentication failed" }
        return json.encode(response)
    end

    -- Dispatch method
    local method = self.api[request.method]
    if not method or type(method) ~= "function" then
        response.error = { code = -32601, message = "Method not found" }
        return json.encode(response)
    end

    -- Call method
    local ok, result = pcall(method, unpack(request.params or {}))
    if not ok then
        response.error = { code = -32000, message = "Server error: " .. tostring(result) }
    else
        response.result = result
    end

    return json.encode(response)
end

function ToolServer:_worker_loop()
    self.context = zmq.context()
    self.socket = self.context:socket(zmq.REP)
    self.socket:bind(self.endpoint)

    print("[ToolServer] Listening on " .. self.endpoint)
    if self.auth_token then
        print("[ToolServer] Authentication is enabled.")
    end
end

function ToolServer:start()
    if self.running then return end
    self.running = true
    self:_worker_loop()
    print("[ToolServer] Started.")
end

function ToolServer:update(dt)
    if not self.running then return end

    local ok, request_str = pcall(self.socket.recv, self.socket, zmq.DONTWAIT)
    if ok and request_str then
        local response_str = self:_handle_request(request_str)
        self.socket:send(response_str)
    end
end

function ToolServer:stop()
    if not self.running then return end
    self.running = false

    if self.socket then
        self.socket:close()
    end
    if self.context then
        self.context:term()
    end
    print("[ToolServer] Stopped.")
end

return ToolServer