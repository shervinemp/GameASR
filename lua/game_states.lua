require "abstractions/scene"

-- Define GameStates
GameState = {}

function GameState.new()
  local state = Scene.new() -- Inherit from Scene abstraction
  setmetatable(state, {__index=Scene})
  return state
end

-- Create all states first to avoid reference errors
local start_state = GameState.new()
start_state.name = "start"

local play_state = GameState.new()
play_state.name = "play"

local over_state = GameState.new()
over_state.name = "over"

-- Setup each state's properties and methods
start_state.enter = function() print("Entering start state") end
start_state.draw = function()
  love.graphics.print("Press any key to start", window.width / 2 - 100, window.height / 2)
end
start_state.keypressed = function(key)
  current_state = "play"
  play_state.enter()
end

play_state.enter = function() print("Entering play state") end
play_state.update = function(dt)
  if type(dt) == "table" then dt = 1/60 end -- Default to 60 FPS frame time if dt is not a number

  -- Directly update game logic instead of recursive call
  input:handleEvent({type="update", dt=dt})

  local y_dir = 1
  if love.keyboard.isDown("down") or love.keyboard.isDown("s") then
    player.x = player.x + player.speed * math.sin(player.angle) * dt
    player.y = player.y - player.speed * math.cos(player.angle) * dt
    y_dir = -1
  end
  if love.keyboard.isDown("up") or love.keyboard.isDown("w") then
    player.x = player.x - player.speed * math.sin(player.angle) * dt
    player.y = player.y + player.speed * math.cos(player.angle) * dt
    y_dir = 1
  end
  if love.keyboard.isDown("left") or love.keyboard.isDown("a") then
    player.angle = (player.angle - math.pi * player.angularSpeed * dt * y_dir) % (2 * math.pi)
  end
  if love.keyboard.isDown("right") or love.keyboard.isDown("d") then
    player.angle = (player.angle + math.pi * player.angularSpeed * dt * y_dir) % (2 * math.pi)
  end

  -- Keep player within bounds using collision system
  if not physics.collisionSystem:isWithinBounds(player, window.width, window.height) then
    -- Bounce back player if out of bounds
    if player.x - player.width / 2 < 0 or player.x + player.width / 2 > window.width then
      player.dx = -player.dx
    end
    if player.y - player.height / 2 < 0 or player.y + player.height / 2 > window.height then
      player.dy = -player.dy
    end

    -- Clamp to screen edges
    player.x = math.max(player.width / 2, math.min(player.x, window.width - player.width / 2))
    player.y = math.max(player.height / 2, math.min(player.y, window.height - player.height / 2))
  end

  -- Now update physics for everything else
  physics:update(dt)
end

play_state.keypressed = function(key)
  if key == "escape" then
    current_state = "start"
    start_state.enter()
  elseif key == "p" then
    current_state = "over"
    over_state.enter()
  end
end

over_state.enter = function() print("Entering over state") end
over_state.draw = function()
  love.graphics.print("Game Over! Press R to restart", window.width / 2 - 100, window.height / 2)
end
over_state.keypressed = function(key)
  if key == "r" then
    current_state = "start"
    start_state.enter()
  end
end

return {
  start = start_state,
  play = play_state,
  over = over_state
}
