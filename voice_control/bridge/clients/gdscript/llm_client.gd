# llm_client.gd
# This script is a client for the Python LLMServer.
# It requires a ZeroMQ addon for Godot (e.g., godot-zeromq).

extends Node

signal response_received(response)

var zmq = ZMQ.new()
var socket
var endpoint = "tcp://0.0.0.0:8000"
var auth_token = null # Set this to your auth token if needed

var _id_counter = 0
var pending_requests = false

func _ready():
    socket = zmq.socket(ZMQ.REQ)
    socket.connect(endpoint)
    print("[LLMClient] Connected to ", endpoint)

    # Example query - now requires yielding or connecting to signal in Godot
    query("Hello from Godot!")
    print("[LLMClient] Query sent, waiting for response via _process")

func _exit_tree():
    socket.close()
    zmq.term()
    print("[LLMClient] Disconnected")

func query(content, role="user"):
    var params = {"content": content, "role": role}
    return _request("query", params)

func _process(delta):
    if pending_requests and socket.poll(0) > 0:
        var response_json = socket.recv_json(ZMQ.DONTWAIT)
        pending_requests = false
        if typeof(response_json) != TYPE_DICTIONARY:
            push_error("LLMClient received non-dictionary JSON response")
            return
        emit_signal("response_received", response_json)

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
    pending_requests = true