#include "tool_server.hpp"
#include <iostream>
#include <thread>

// --- Dummy Tool Implementations ---
json ToolServer::move_player(const json &params)
{
    std::string direction = params.value("direction", "unknown");
    std::cout << "[ToolServer] Moving player: " << direction << std::endl;
    return {{"status", "success"}, {"message", "Moved " + direction}};
}

json ToolServer::get_player_position(const json &params)
{
    std::cout << "[ToolServer] Getting player position" << std::endl;
    return {{"x", 300}, {"y", 400}};
}

json ToolServer::set_game_pause(const json &params)
{
    bool is_paused = params.value("is_paused", false);
    std::cout << "[ToolServer] Setting game pause state to: " << std::boolalpha << is_paused << std::endl;
    return {{"status", "success"}};
}

json ToolServer::get_game_time(const json &params)
{
    std::cout << "[ToolServer] Getting game time" << std.endl;
    return {{"time", "5:00 PM"}};
}

// --- Server Implementation ---
ToolServer::ToolServer(const std::string &endpoint, const std::string &authToken)
    : _context(1), _socket(_context, zmq::socket_type::rep), _endpoint(endpoint), _authToken(authToken), _isRunning(false)
{
    _rpcMethods["move_player"] = move_player;
    _rpcMethods["get_player_position"] = get_player_position;
    _rpcMethods["set_game_pause"] = set_game_pause;
    _rpcMethods["get_game_time"] = get_game_time;
}

ToolServer::~ToolServer()
{
    stop();
}

void ToolServer::start()
{
    if (_isRunning)
        return;
    _isRunning = true;
    _serverThread = std::thread(&ToolServer::server_loop, this);
    std::cout << "[ToolServer] Listening on " << _endpoint << std::endl;
    if (!_authToken.empty())
    {
        std::cout << "[ToolServer] Authentication is enabled." << std::endl;
    }
}

void ToolServer::stop()
{
    _isRunning = false;
    if (_serverThread.joinable())
    {
        _serverThread.join();
    }
    std::cout << "[ToolServer] Stopped." << std::endl;
}

void ToolServer::server_loop()
{
    _socket.bind(_endpoint);
    while (_isRunning)
    {
        zmq::pollitem_t items[] = {
            { static_cast<void*>(_socket), 0, ZMQ_POLLIN, 0 }
        };
        zmq::poll(items, 1, std::chrono::milliseconds(100));

        if (items[0].revents & ZMQ_POLLIN)
        {
            zmq::message_t request_msg;
            if (_socket.recv(request_msg, zmq::recv_flags::none))
            {
                json request = json::parse(request_msg.to_string());
                json response = handle_request(request);
                _socket.send(zmq::buffer(response.dump()), zmq::send_flags::none);
            }
        }
    }
}

json ToolServer::handle_request(const json &request)
{
    json response;
    try
    {
        if (!_authToken.empty() && (!request.contains("auth_token") || request["auth_token"] != _authToken))
        {
            throw std::runtime_error("Authentication failed");
        }

        std::string method_name = request.at("method");
        if (_rpcMethods.find(method_name) == _rpcMethods.end())
        {
            throw std::runtime_error("Method not found");
        }

        json result = _rpcMethods[method_name](request.value("params", json::object()));
        response = {
            {"jsonrpc", "2.0"},
            {"result", result},
            {"id", request.value("id", nullptr)}};
    }
    catch (const std::exception &e)
    {
        response = {
            {"jsonrpc", "2.0"},
            {"error", {{"code", -32000}, {"message", e.what()}}},
            {"id", request.value("id", nullptr)}};
    }
    return response;
}

// Example Usage
int main()
{
    const char *token = std::getenv("TOOLS_AUTH_TOKEN");
    ToolServer server("tcp://*:8080", token ? std::string(token) : "");
    server.start();

    std::cout << "ToolServer running. Press Enter to stop." << std::endl;
    std::cin.get();

    server.stop();
    return 0;
}