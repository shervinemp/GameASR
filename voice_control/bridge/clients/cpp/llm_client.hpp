#ifndef LLM_CLIENT_HPP
#define LLM_CLIENT_HPP

#include <string>
#include <zmq.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class LLMClient
{
public:
    LLMClient(const std::string &endpoint = "tcp://127.0.0.1:8000", const std::string &authToken = "");
    ~LLMClient();

    void connect();
    void disconnect();
    json query(const std::string &content, const std::string &role = "user");

private:
    json _request(const std::string &method, const json &params);

    zmq::context_t _context;
    zmq::socket_t _socket;
    std::string _endpoint;
    std::string _authToken;
    int _idCounter;
};

#endif // LLM_CLIENT_HPP