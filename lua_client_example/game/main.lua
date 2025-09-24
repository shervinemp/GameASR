package.path = package.path .. ";./lua/?.lua;./lua/?/init.lua"

require "abstractions.scene"
require "abstractions.input"
require "abstractions.physics"
local game_states = require("game_states")
local rpc_api = require("rpc_api")
local ToolServer = require("voice_control.bridge.clients.lua.tool_server")

-- Main game loop

function draw_scene()
  love.graphics.clear(1, 1, 1)
  if player and player.sprite then
    local sprite_width = player.sprite:getWidth()
    local sprite_height = player.sprite:getHeight()
    love.graphics.draw(player.sprite, player.x, player.y, player.angle, 1, 1,
                       sprite_width / 2, sprite_height / 2)
  end
end

function love.load()

  current_state = "start"
  start_time = love.timer.getTime()
  window = {
    width = 1200,
    height = 800,
  }
  love.window.setMode(window.width, window.height)

  -- Initialize abstractions
  input = Input.new()
  physics = Physics.new()

  -- Create player object using abstractions and initialize global player object for all states
  local success, err = pcall(function()
    local sprite = love.graphics.newImage("assets/player.png")
    if sprite then
      player = {
        x = 600,
        y = 400,
        angle = 0,
        speed = 50,  -- Increased for better control in top-down shooter
        angularSpeed = 0.25,  -- Adjusted for top-down shooter
        dx = 0,
        dy = 0,
        sprite = sprite,
        width = sprite:getWidth(),
        height = sprite:getHeight()
      }
    else
      print("Error: Could not load player sprite")
      return nil
    end
  end)

  if not success then
    print("Error loading player sprite:", err)
    return
  end

  -- Register input handlers for top-down movement
  input:registerKey("up", function() player.dy = -player.speed end)
  input:registerKey("down", function() player.dy = player.speed end)
  input:registerKey("left", function() player.dx = -player.speed end)
  input:registerKey("right", function() player.dx = player.speed end)

  -- Add player to physics
  physics:addBody(player)

  -- Load game states after everything is initialized
  local success, err = pcall(function()
    game_states = require("game_states")
    game_states.start.enter()
  end)

  if not success then
    print("Error loading game states:", err)
    return
  end

  if not success then
    print("Error loading game states:", err)
    return
  end

  -- Start the tool server
  local endpoint = os.getenv("LUA_TOOLS_ENDPOINT") or "tcp://127.0.0.1:8080"
  local auth_token = os.getenv("LUA_TOOLS_AUTH_TOKEN")
  tool_server = ToolServer:new(rpc_api, endpoint, auth_token)
  tool_server:start()
end

function love.quit()
    if tool_server then
        tool_server:stop()
    end
end

function love.update(dt)
  if current_state == "play" and player then
    -- First handle custom movement to ensure it takes precedence
    input:handleEvent({type="update", dt=dt})

    -- Apply player's velocity directly (if any) before physics update
    if player.dx ~= 0 or player.dy ~= 0 then
      local angle = math.atan2(player.dy, player.dx)
      player.x = player.x + math.cos(angle) * player.speed * dt
      player.y = player.y + math.sin(angle) * player.speed * dt

      -- Reset velocities for next frame
      if input:isKeyReleased("up") or input:isKeyReleased("down") then
        player.dy = 0
      end
      if input:isKeyReleased("left") or input:isKeyReleased("right") then
        player.dx = 0
      end
    end

    -- Then update physics for all bodies (including player)
    physics:update(dt)

    game_states.play:update(dt)
  end
end

function love.draw()
  if current_state == "start" then
    game_states.start:draw()
  elseif current_state == "play" then
    draw_scene()
  elseif current_state == "over" then
    game_states.over:draw()
  end
end

function love.keypressed(key)
  input:handleEvent({type="keypress", key=key})
  if current_state == "start" then
    game_states.start.keypressed(key)
  elseif current_state == "play" then
    game_states.play.keypressed(key)
  elseif current_state == "over" then
    game_states.over.keypressed(key)
  end
end
