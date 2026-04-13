# llm_client.gd
# This script is a client for the Python LLMServer.
# It requires a ZeroMQ addon for Godot (e.g., godot-zeromq).

extends Node

var zmq = ZMQ.new()
var socket
var endpoint = "tcp://127.0.0.1:8000"
var auth_token = null # Set this to your auth token if needed

var _id_counter = 0

func _ready():
    socket = zmq.socket(ZMQ.REQ)
    socket.connect(endpoint)
    print("[LLMClient] Connected to ", endpoint)

    # Example query
    var response = query("Hello from Godot!")
    if response.error:
        print("Error: ", response.error)
    else:
        print("Response from LLM: ", response.result)

func _exit_tree():
    socket.close()
    zmq.term()
    print("[LLMClient] Disconnected")

func query(content, role="user"):
    var params = {"content": content, "role": role}
    return _request("query", params)

func _request(method, params):
    _id_counter += 1
    var request_body = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": _id_counter
    }
    if auth_token:
        request_body["auth_token"] = auth_token

    socket.send_json(request_body)

    # Non-blocking poll to prevent freezing the Godot engine
    while socket.poll(0) == 0:
        OS.delay_msec(1)

    var response_json = socket.recv_json(ZMQ.DONTWAIT)

    return response_json