#include "llm_client.hpp"
#include <iostream>

LLMClient::LLMClient(const std::string& endpoint, const std::string& authToken)
    : _context(1), _socket(_context, zmq::socket_type::req), _endpoint(endpoint), _authToken(authToken), _idCounter(0) {
}

LLMClient::~LLMClient() {
    disconnect();
}

void LLMClient::connect() {
    _socket.connect(_endpoint);
    std::cout << "[LLMClient] Connected to " << _endpoint << std::endl;
}

void LLMClient::disconnect() {
    _socket.close();
    std::cout << "[LLMClient] Disconnected" << std::endl;
}

json LLMClient::query(const std::string& content, const std::string& role) {
    json params = {
        {"content", content},
        {"role", role}
    };
    return _request("query", params);
}

json LLMClient::_request(const std::string& method, const json& params) {
    _idCounter++;
    json request = {
        {"jsonrpc", "2.0"},
        {"method", method},
        {"params", params},
        {"id", _idCounter}
    };

    if (!_authToken.empty()) {
        request["auth_token"] = _authToken;
    }

    _socket.send(zmq::buffer(request.dump()), zmq::send_flags::none);

    zmq::message_t reply;
    auto received = _socket.recv(reply, zmq::recv_flags::none);

    if (!received.has_value()) {
        throw std::runtime_error("No response from server.");
    }

    auto response = json::parse(reply.to_string());
    if (response.contains("error")) {
        throw std::runtime_error("RPC Error: " + response["error"].dump());
    }

    return response["result"];
}

// Example Usage
// You would need to link against zmq and nlohmann_json libraries.
// g++ -std=c++17 llm_client.cpp -o example -lzmq -lnlohmann_json
int main() {
    try {
        LLMClient client;
        client.connect();
        json response = client.query("Hello from C++!");
        std::cout << "Response from LLM: " << response.dump(4) << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}