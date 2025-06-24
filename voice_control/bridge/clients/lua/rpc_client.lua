-- rpc_client.lua
local json = require("json")
local zmq = require("lzmq")

local VoiceControlClient = {}

--- Constructor for the VoiceControlClient.
-- @param config table A table containing connection details.
--   For TCP: {protocol = "tcp", host = "127.0.0.1", port = 5555}
--   For IPC: {protocol = "ipc", path = "/tmp/voice_control.ipc"}
-- @return VoiceControlClient A new client instance.
function VoiceControlClient:new(config)
    local obj = {
        config = config,
        endpoint = "", -- Will be constructed in connect()
        id_counter = 0,
        context = nil,
        socket = nil
    }
    setmetatable(obj, self)
    self.__index = self
    return obj
end

function VoiceControlClient:connect()
    if self.socket then
        self:disconnect()
    end

    -- Construct the ZeroMQ endpoint string based on config
    if self.config.protocol == "tcp" then
        if not self.config.host or not self.config.port then
            return false, "TCP protocol requires 'host' and 'port' in config."
        end
        self.endpoint = string.format("tcp://%s:%d", self.config.host, self.config.port)
    elseif self.config.protocol == "ipc" then
        if not self.config.path then
            return false, "IPC protocol requires 'path' in config."
        end
        self.endpoint = string.format("ipc://%s", self.config.path)
    else
        return false, "Unsupported protocol: " .. tostring(self.config.protocol)
    end

    local success, err_msg
    
    success, err_msg = pcall(function() self.context = zmq.context() end)
    if not success then return false, "Failed to create ZeroMQ context: " .. tostring(err_msg) end

    success, err_msg = pcall(function() self.socket = self.context:socket(zmq.REQ) end)
    if not success then
        self.context:term()
        self.context = nil
        return false, "Failed to create ZeroMQ REQ socket: " .. tostring(err_msg)
    end
    
    self.socket:set_option(zmq.RCVTIMEO, 5000) -- 5 seconds receive timeout
    self.socket:set_option(zmq.SNDTIMEO, 5000) -- 5 seconds send timeout
    
    local connected_ok, connect_err = pcall(function()
        self.socket:connect(self.endpoint)
    end)

    if not connected_ok then
        self.socket:close()
        self.context:term()
        self.socket = nil
        self.context = nil
        return false, "Failed to connect to ZeroMQ: " .. tostring(connect_err)
    end

    print("[Lua-Client] Connected to endpoint:", self.endpoint) -- Added for clarity
    return true
end

function VoiceControlClient:disconnect()
    if self.socket then self.socket:close() end
    if self.context then self.context:term() end
    self.socket = nil
    self.context = nil
    print("[Lua-Client] Disconnected.") -- Added for clarity
end

function VoiceControlClient:_request(method, params)
    if not self.socket then
        return nil, "Not connected to ZeroMQ. Call connect() first."
    end

    self.id_counter = self.id_counter + 1
    local request_body = {
        jsonrpc = "2.0",
        method = method,
        params = params or {},
        id = self.id_counter
    }

    local encoded_request = json.encode(request_body)

    local send_success, send_result_or_err = pcall(function()
        return self.socket:send(encoded_request)
    end)

    if not send_success then
        self:disconnect()
        local reconnected, recon_err = self:connect()
        if not reconnected then
            return nil, "Failed to send & re-establish connection: " .. tostring(send_result_or_err) .. " | " .. recon_err
        end
        return nil, "Failed to send ZeroMQ message: " .. tostring(send_result_or_err) .. " (socket reconnected)"
    end

    local recv_success, recv_data = pcall(function()
        return self.socket:recv()
    end)

    if not recv_success then
        if recv_data == "timeout" then
            self:disconnect()
            local reconnected, recon_err = self:connect()
            if not reconnected then
                return nil, "ZeroMQ receive timeout. Could not re-establish connection: " .. recon_err
            end
            return nil, "ZeroMQ receive timeout. Socket recovered for future requests."
        end
        return nil, "Failed to receive ZeroMQ message: " .. tostring(recv_data)
    end

    local response_str = recv_data

    if not response_str or #response_str == 0 then
        return nil, "Empty response from Python server via ZeroMQ."
    end

    local decoded_response
    local json_decode_success, err_msg = pcall(function() decoded_response = json.decode(response_str) end)
    if not json_decode_success then
        return nil, "Failed to decode JSON response: " .. err_msg .. "\nBody: " .. response_str
    end

    if decoded_response.jsonrpc ~= "2.0" then
        return nil, "Invalid JSON-RPC version in response."
    end

    if decoded_response.error then
        local err_msg = decoded_response.error.message or "Unknown error"
        local err_code = decoded_response.error.code or -1
        return nil, string.format("Python RPC Error (Code %d): %s", err_code, err_msg)
    end

    if decoded_response.id ~= request_body.id then
        return nil, "Mismatched RPC response ID. Expected " .. request_body.id .. ", got " .. tostring(decoded_response.id or "nil")
    end

    if decoded_response.result ~= nil then
        return decoded_response.result, nil
    else
        return nil, "Missing 'result' or 'error' field in JSON-RPC response."
    end
end

function VoiceControlClient:setContexts(contexts)
    print("[Lua-Client] Sending set_contexts...")
    local result, err = self:_request("set_contexts", {contexts = contexts})
    if err then print("[Lua-Client] Error setting contexts:", err) end
    return result, err
end

function VoiceControlClient:query(prompt)
    print("[Lua-Client] Sending query:", prompt)
    local result, err = self:_request("query", {prompt = prompt})
    if err then print("[Lua-Client] Error querying:", err) end
    return result, err
end

return VoiceControlClient