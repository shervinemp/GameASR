// tool_server.js
// Requires the 'zeromq' npm package.
// This is a server that exposes application functions to the Python client.

const zmq = require("zeromq");

// --- Dummy Tool Implementations ---
function movePlayer(params) {
    console.log(`[ToolServer] Moving player: ${params.direction}`);
    return { status: "success", message: `Moved ${params.direction}` };
}

function getPlayerPosition(params) {
    console.log("[ToolServer] Getting player position");
    return { x: 50, y: 60 };
}

function setGamePause(params) {
    console.log(`[ToolServer] Setting game pause state to: ${params.is_paused}`);
    return { status: "success" };
}

function getGameTime(params) {
    console.log("[ToolServer] Getting game time");
    return { time: "10:00 AM" };
}

// --- RPC Method Dispatcher ---
const RPC_METHODS = {
    "move_player": movePlayer,
    "get_player_position": getPlayerPosition,
    "set_game_pause": setGamePause,
    "get_game_time": getGameTime,
};

class ToolServer {
    constructor(endpoint = "tcp://0.0.0.0:8080", authToken = null) {
        this.endpoint = endpoint;
        this.authToken = authToken;
        this.socket = new zmq.Reply();
        this.isRunning = false;
    }

    async start() {
        await this.socket.bind(this.endpoint);
        this.isRunning = true;
        console.log(`[ToolServer] Listening on ${this.endpoint}`);
        if (this.authToken) {
            console.log("[ToolServer] Authentication is enabled.");
        }

        while (this.isRunning) {
            try {
                const [msg] = await this.socket.receive();
                const request = JSON.parse(msg.toString());
                const response = this._handleRequest(request);
                await this.socket.send(JSON.stringify(response));
            } catch (e) {
                if (!this.isRunning) break;
                console.error("ZMQ receive error:", e);
            }
        }
    }

    stop() {
        this.isRunning = false;
        this.socket.close();
        console.log("[ToolServer] Stopped.");
    }

    _handleRequest(request) {
        try {
            if (this.authToken && request.auth_token !== this.authToken) {
                throw new Error("Authentication failed");
            }

            const method = RPC_METHODS[request.method];
            if (!method) {
                throw new Error("Method not found");
            }

            const result = method(request.params || {});
            return { jsonrpc: "2.0", result: result, id: request.id };

        } catch (e) {
            return {
                jsonrpc: "2.0",
                error: { code: -32000, message: e.message },
                id: request.id,
            };
        }
    }
}

// Example Usage
async function main() {
    const authToken = process.env.TOOLS_AUTH_TOKEN || null;
    const server = new ToolServer(undefined, authToken);

    console.log("ToolServer running. Press Ctrl+C to stop.");
    await server.start();
}

if (require.main === module) {
    main();
}

module.exports = ToolServer;