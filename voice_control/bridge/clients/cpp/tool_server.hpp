#ifndef TOOL_SERVER_HPP
#define TOOL_SERVER_HPP

#include <string>
#include <map>
#include <functional>
#include <zmq.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class ToolServer
{
public:
    ToolServer(const std::string &endpoint = "tcp://127.0.0.1:8080", const std::string &authToken = "");
    ~ToolServer();

    void start();
    void stop();

private:
    void server_loop();
    json handle_request(const json &request);

    // --- Dummy Tool Implementations ---
    static json move_player(const json &params);
    static json get_player_position(const json &params);
    static json set_game_pause(const json &params);
    static json get_game_time(const json &params);

    zmq::context_t _context;
    zmq::socket_t _socket;
    std::string _endpoint;
    std::string _authToken;
    bool _isRunning;
    std::thread _serverThread;

    using RpcMethod = std::function<json(const json &)>;
    std::map<std::string, RpcMethod> _rpcMethods;
};

#endif // TOOL_SERVER_HPP