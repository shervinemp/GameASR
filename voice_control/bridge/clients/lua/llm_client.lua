-- voice_control_api.lua
-- llm_client.lua
-- This file defines the RPC client for making calls from a Lua application
-- to the Python LLM server.

local json = require("json")
local zmq = require("lzmq")

local LLMClient = {}
LLMClient.__index = LLMClient

--- Constructor for the LLMClient.
-- @param protocol string The connection protocol ("tcp" or "ipc"). Defaults to "tcp".
-- @param endpoint string The endpoint string.
-- @return LLMClient A new API client instance.
function LLMClient:new(protocol, endpoint)
    protocol = protocol or "tcp"

    if protocol == "tcp" then
        endpoint = endpoint or "0.0.0.0:8000" -- Default for the LLM Server
    end

    local obj = setmetatable({
        protocol = protocol,
        endpoint = endpoint,
        zmq_full_endpoint = "",
        id_counter = 0,
        context = nil,
        socket = nil
    }, self)
    return obj
end

function LLMClient:connect()
    if self.socket then self:disconnect() end

    if self.protocol == "tcp" then
        self.zmq_full_endpoint = string.format("tcp://%s", self.endpoint)
    elseif self.protocol == "ipc" then
        self.zmq_full_endpoint = string.format("ipc://%s", self.endpoint)
    else
        return false, "Unsupported protocol: " .. tostring(self.protocol)
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

    self.socket:set_option(zmq.RCVTIMEO, 30000)
    self.socket:set_option(zmq.SNDTIMEO, 5000)

    local connected_ok, connect_err = pcall(function()
        self.socket:connect(self.zmq_full_endpoint)
    end)

    if not connected_ok then
        self.socket:close()
        self.context:term()
        self.socket = nil
        self.context = nil
        return false, "Failed to connect to ZeroMQ: " .. tostring(connect_err)
    end

    print("[LLM-Client] Connected to endpoint:", self.zmq_full_endpoint)
    return true
end

function LLMClient:disconnect()
    if self.socket then self.socket:close() end
    if self.context then self.context:term() end
    self.socket = nil
    self.context = nil
    print("[LLM-Client] Disconnected.")
end

function LLMClient:_request(method, params)
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
        return nil, "Failed to send ZeroMQ message: " .. tostring(send_result_or_err)
    end

    local recv_success, recv_data = pcall(function()
        return self.socket:recv()
    end)

    if not recv_success then
        return nil, "Failed to receive ZeroMQ message: " .. tostring(recv_data)
    end

    local response_str = recv_data
    if not response_str or #response_str == 0 then
        return nil, "Empty response from server."
    end

    local decoded_response
    local json_decode_success, err_msg = pcall(function() decoded_response = json.decode(response_str) end)
    if not json_decode_success then
        return nil, "Failed to decode JSON response: " .. err_msg
    end

    if decoded_response.error then
        return nil, string.format("RPC Error (Code %d): %s", decoded_response.error.code, decoded_response.error.message)
    end

    return decoded_response.result, nil
end

function LLMClient:query(content, role)
    role = role or "user"
    if not self.socket then
        return nil, "Not connected to ZeroMQ. Call connect() first."
    end

    self.id_counter = self.id_counter + 1
    local request_body = {
        jsonrpc = "2.0",
        method = "query",
        params = {role = role, content = content},
        id = self.id_counter
    }

    local encoded_request = json.encode(request_body)

    local send_success, send_result_or_err = pcall(function()
        return self.socket:send(encoded_request)
    end)

    if not send_success then
        return nil, "Failed to send ZeroMQ message: " .. tostring(send_result_or_err)
    end

    return true, nil
end

function LLMClient:poll()
    if not self.socket then
        return nil, "Not connected"
    end

    local recv_success, recv_data = pcall(function()
        return self.socket:recv(zmq.DONTWAIT)
    end)

    if not recv_success or not recv_data then
        return nil
    end

    local decoded_response
    local json_decode_success, _ = pcall(function() decoded_response = json.decode(recv_data) end)
    if not json_decode_success then
        return nil
    end

    return decoded_response.result or decoded_response
end

return LLMClient