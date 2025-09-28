// tool_server.cs
// Requires the NetMQ NuGet package.
// This is a server that exposes application functions to the Python client.

using System;
using System.Collections.Generic;
using System.Threading;
using NetMQ;
using NetMQ.Sockets;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

public class ToolServer : IDisposable
{
    private ResponseSocket _server;
    private readonly string _endpoint;
    private readonly string _authToken;
    private bool _isRunning = false;
    private Thread _serverThread;

    // --- RPC Method Dispatcher ---
    private readonly Dictionary<string, Func<JObject, JObject>> _rpcMethods;

    public ToolServer(string endpoint = "tcp://127.0.0.1:8080", string authToken = null)
    {
        _endpoint = endpoint;
        _authToken = authToken;
        _rpcMethods = new Dictionary<string, Func<JObject, JObject>>
        {
            {"move_player", MovePlayer},
            {"get_player_position", GetPlayerPosition},
            {"set_game_pause", SetGamePause},
            {"get_game_time", GetGameTime}
        };
    }

    #region Dummy Tool Implementations
    private JObject MovePlayer(JObject parameters)
    {
        var direction = parameters["direction"]?.ToString() ?? "unknown";
        Console.WriteLine($"[ToolServer] Moving player: {direction}");
        return new JObject {["status"] = "success", ["message"] = $"Moved {direction}"};
    }

    private JObject GetPlayerPosition(JObject parameters)
    {
        Console.WriteLine("[ToolServer] Getting player position");
        return new JObject {["x"] = 150, ["y"] = 250};
    }

    private JObject SetGamePause(JObject parameters)
    {
        var isPaused = parameters["is_paused"]?.ToObject<bool>() ?? false;
        Console.WriteLine($"[ToolServer] Setting game pause state to: {isPaused}");
        return new JObject {["status"] = "success"};
    }

    private JObject GetGameTime(JObject parameters)
    {
        Console.WriteLine("[ToolServer] Getting game time");
        return new JObject {["time"] = "3:00 PM"};
    }
    #endregion

    public void Start()
    {
        if (_isRunning) return;
        _isRunning = true;
        _serverThread = new Thread(ServerLoop);
        _serverThread.Start();
        Console.WriteLine($"[ToolServer] Listening on {_endpoint}");
        if (!string.IsNullOrEmpty(_authToken))
        {
            Console.WriteLine("[ToolServer] Authentication is enabled.");
        }
    }

    public void Stop()
    {
        _isRunning = false;
        // In a real app, you might need a more graceful shutdown
        if (_serverThread != null && _serverThread.IsAlive)
        {
            // NetMQ sockets can be tricky to close from another thread.
            // In a real Unity/Godot app, you'd integrate this into the main loop
            // or use async patterns instead of a blocking thread.
        }
        Console.WriteLine("[ToolServer] Stopped.");
    }

    private void ServerLoop()
    {
        using (var server = new ResponseSocket())
        {
            server.Bind(_endpoint);
            while (_isRunning)
            {
                if (server.TryReceiveFrameString(TimeSpan.FromSeconds(1), out var message))
                {
                    var response = _HandleRequest(message);
                    server.SendFrame(response);
                }
            }
        }
    }

    private string _HandleRequest(string requestJson)
    {
        JObject response;
        JObject request = null;
        try
        {
            request = JObject.Parse(requestJson);

            if (!string.IsNullOrEmpty(_authToken) && request["auth_token"]?.ToString() != _authToken)
            {
                throw new Exception("Authentication failed");
            }

            var methodName = request["method"]?.ToString();
            if (string.IsNullOrEmpty(methodName) || !_rpcMethods.ContainsKey(methodName))
            {
                throw new Exception("Method not found");
            }

            var result = _rpcMethods[methodName](request["params"] as JObject);
            response = new JObject
            {
                ["jsonrpc"] = "2.0",
                ["result"] = result,
                ["id"] = request["id"]
            };
        }
        catch (Exception ex)
        {
            response = new JObject
            {
                ["jsonrpc"] = "2.0",
                ["error"] = new JObject {["code"] = -32000, ["message"] = ex.Message},
                ["id"] = request?["id"]
            };
        }
        return response.ToString(Formatting.None);
    }

    public void Dispose()
    {
        Stop();
    }

    // Example Usage
    public static void Main(string[] args)
    {
        var authToken = Environment.GetEnvironmentVariable("TOOLS_AUTH_TOKEN");
        using (var server = new ToolServer(authToken: authToken))
        {
            server.Start();
            Console.WriteLine("ToolServer running. Press Enter to stop.");
            Console.ReadLine();
        }
    }
}