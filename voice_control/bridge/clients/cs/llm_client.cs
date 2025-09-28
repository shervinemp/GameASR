// llm_client.cs
// Requires the NetMQ NuGet package.
// This is a client for the Python LLMServer.

using System;
using NetMQ;
using NetMQ.Sockets;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System.Threading;

public class LLMClient : IDisposable
{
    private RequestSocket _client;
    private readonly string _endpoint;
    private readonly string _authToken;
    private int _idCounter = 0;

    public LLMClient(string endpoint = "tcp://127.0.0.1:8000", string authToken = null)
    {
        _endpoint = endpoint;
        _authToken = authToken;
    }

    public void Connect()
    {
        _client = new RequestSocket();
        _client.Connect(_endpoint);
        Console.WriteLine($"[LLMClient] Connected to {_endpoint}");
    }

    public void Disconnect()
    {
        if (_client != null)
        {
            _client.Dispose();
            _client = null;
        }
        Console.WriteLine("[LLMClient] Disconnected");
    }

    public JObject Query(string content, string role = "user")
    {
        var parameters = new JObject
        {
            ["content"] = content,
            ["role"] = role
        };
        return _Request("query", parameters);
    }

    private JObject _Request(string method, JObject parameters)
    {
        _idCounter++;
        var request = new JObject
        {
            ["jsonrpc"] = "2.0",
            ["method"] = method,
            ["params"] = parameters,
            ["id"] = _idCounter
        };

        if (!string.IsNullOrEmpty(_authToken))
        {
            request["auth_token"] = _authToken;
        }

        _client.SendFrame(request.ToString(Formatting.None));

        if (_client.TryReceiveFrameString(TimeSpan.FromSeconds(5), out var responseJson))
        {
            var response = JObject.Parse(responseJson);
            if (response["error"] != null)
            {
                throw new Exception($"RPC Error: {response["error"]}");
            }
            return response["result"] as JObject;
        }
        else
        {
            throw new TimeoutException("No response from server within the timeout period.");
        }
    }

    public void Dispose()
    {
        Disconnect();
    }

    // Example Usage
    public static void Main(string[] args)
    {
        using (var client = new LLMClient())
        {
            client.Connect();
            try
            {
                var response = client.Query("Hello from C#!");
                Console.WriteLine("Response from LLM: " + response);
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error: " + ex.Message);
            }
        }
    }
}