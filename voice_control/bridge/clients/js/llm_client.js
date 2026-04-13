// llm_client.js
// Requires the 'zeromq' npm package.
// This is a client for the Python LLMServer.

const zmq = require("zeromq");

class LLMClient {
    constructor(endpoint = "tcp://0.0.0.0:8000", authToken = null) {
        this.endpoint = endpoint;
        this.authToken = authToken;
        this.socket = new zmq.Request();
        this._idCounter = 0;
    }

    async connect() {
        this.socket.connect(this.endpoint);
        console.log(`[LLMClient] Connected to ${this.endpoint}`);
    }

    async disconnect() {
        this.socket.close();
        console.log("[LLMClient] Disconnected");
    }

    async query(content, role = "user") {
        const params = { content, role };
        return this._request("query", params);
    }

    async _request(method, params) {
        this._idCounter++;
        const request = {
            jsonrpc: "2.0",
            method: method,
            params: params,
            id: this._idCounter,
        };

        if (this.authToken) {
            request.auth_token = this.authToken;
        }

        await this.socket.send(JSON.stringify(request));
        const [result] = await this.socket.receive();
        const response = JSON.parse(result.toString());

        if (response.error) {
            throw new Error(`RPC Error: ${JSON.stringify(response.error)}`);
        }
        return response.result;
    }
}

// Example Usage
async function main() {
    const client = new LLMClient();
    await client.connect();
    try {
        const response = await client.query("Hello from Node.js!");
        console.log("Response from LLM:", response);
    } catch (e) {
        console.error("Error:", e.message);
    } finally {
        await client.disconnect();
    }
}

if (require.main === module) {
    main();
}

module.exports = LLMClient;