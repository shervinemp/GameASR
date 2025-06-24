-- File: lua_client_example/game/lua/abstractions/__init__.lua
-- Initializes the abstractions module

local abstractions = {}

function abstractions.load_all()
    -- Load all abstraction modules
    abstractions.audio = require("lua.abstractions.audio")
    abstractions.collision = require("lua.abstractions.collision")
    abstractions.input = require("lua.abstractions.input")
    abstractions.io = require("lua.abstractions.io")
    abstractions.physics = require("lua.abstractions.physics")
    abstractions.rendering = require("lua.abstractions.rendering")
    abstractions.scene = require("lua.abstractions.scene")
    abstractions.ui = require("lua.abstractions.ui")
end

return abstractions