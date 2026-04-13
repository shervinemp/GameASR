# tool_server.gd
# This script is a server that exposes game functions to the Python application.
# It requires a ZeroMQ addon for Godot (e.g., godot-zeromq).

extends Node

var zmq = ZMQ.new()
var socket
var endpoint = "tcp://0.0.0.0:8080"
var auth_token = null # Set this via environment variable or config

# --- RPC Method Dispatcher ---
var RPC_METHODS = {
    "move_player": funcref(self, "_move_player"),
    "get_player_position": funcref(self, "_get_player_position"),
    "set_game_pause": funcref(self, "_set_game_pause"),
    "get_game_time": funcref(self, "_get_game_time"),
}

# --- Dummy Tool Implementations ---
func _move_player(params):
    var direction = params.get("direction", "unknown")
    print("[ToolServer] Moving player: ", direction)
    return {"status": "success", "message": "Moved " + direction}

func _get_player_position(params):
    print("[ToolServer] Getting player position")
    return {"x": 100, "y": 200}

func _set_game_pause(params):
    var is_paused = params.get("is_paused", false)
    print("[ToolServer] Setting game pause state to: ", is_paused)
    return {"status": "success"}

func _get_game_time(params):
    print("[ToolServer] Getting game time")
    return {"time": "1:00 PM"}

# --- Server Loop ---
func _ready():
    socket = zmq.socket(ZMQ.REP)
    socket.bind(endpoint)
    print("[ToolServer] Listening on ", endpoint)
    if auth_token:
        print("[ToolServer] Authentication is enabled.")

    # Run the server loop in a separate thread
    var thread = Thread.new()
    thread.start(self, "_server_loop")

func _server_loop(_userdata):
    while true:
        var message = socket.recv_json()
        var response = _handle_request(message)
        socket.send_json(response)

func _handle_request(request):
    if auth_token and request.get("auth_token") != auth_token:
        return {"jsonrpc": "2.0", "error": {"code": -32001, "message": "Authentication failed"}, "id": request.get("id")}

    var method_name = request.get("method")
    var params = request.get("params", {})

    var method = RPC_METHODS.get(method_name)
    if not method:
        return {"jsonrpc": "2.0", "error": {"code": -32601, "message": "Method not found"}, "id": request.get("id")}

    var result = method.call_func(params)
    return {"jsonrpc": "2.0", "result": result, "id": request.get("id")}

func _exit_tree():
    socket.close()
    zmq.term()
    print("[ToolServer] Stopped.")