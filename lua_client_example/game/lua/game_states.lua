-- File: lua_client_example/game/lua/game_states.lua
local GameState = {}
GameState.states = {}
GameState.current = {}

-- A simple class for any controllable unit (player or teammate)
local Unit = {}
Unit.__index = Unit

function Unit.new(name, x, y, color)
    local unit = setmetatable({}, Unit)
    unit.name = name
    unit.x = x
    unit.y = y
    unit.target_x = x -- Movement target
    unit.target_y = y
    unit.angle = 0
    unit.speed = 150
    unit.color = color or {1, 1, 1}
    unit.size = 15
    unit.gun_length = 20
    return unit
end

function Unit:update(dt)
    -- Move towards the target position if it exists
    local dx = self.target_x - self.x
    local dy = self.target_y - self.y
    local dist = math.sqrt(dx*dx + dy*dy)

    if dist > 0 then
        local move_dist = self.speed * dt
        if dist <= move_dist then
            self.x = self.target_x
            self.y = self.target_y
        else
            self.x = self.x + (dx / dist) * move_dist
            self.y = self.y + (dy / dist) * move_dist
        end
    end
end

function Unit:shoot_at(tx, ty)
    -- Aim at the target
    self.angle = math.atan2(ty - self.y, tx - self.x)
    -- Create a bullet traveling in that direction
    GameState.current:spawn_bullet(
        self.x + self.gun_length * math.cos(self.angle),
        self.y + self.gun_length * math.sin(self.angle),
        self.angle
    )
end

function Unit:draw()
    -- Draw body
    love.graphics.setColor(self.color)
    love.graphics.circle("fill", self.x, self.y, self.size)

    -- Draw gun/orientation line
    love.graphics.setLineWidth(3)
    love.graphics.line(self.x, self.y, self.x + self.gun_length * math.cos(self.angle), self.y + self.gun_length * math.sin(self.angle))

    -- Draw name
    love.graphics.setColor(1, 1, 1)
    love.graphics.print(self.name, self.x - 15, self.y - 30)
end

-- Main Play State
local play_state = {}

function play_state:enter()
    self.squad = {}
    self.squad["player"] = Unit.new("player", 100, 300, {0, 1, 0}) -- Green
    self.squad["alpha"] = Unit.new("alpha", 150, 250, {0, 0, 1}) -- Blue
    self.squad["bravo"] = Unit.new("bravo", 150, 350, {1, 0, 0}) -- Red

    self.bullets = {}
    self.bullet_speed = 500
    self.bullet_lifetime = 2
end

function play_state:update(dt)
    -- Update player rotation to face the mouse
    local player = self.squad["player"]
    player.angle = math.atan2(love.mouse.getY() - player.y, love.mouse.getX() - player.x)

    -- Update all units
    for name, unit in pairs(self.squad) do
        unit:update(dt)
    end

    -- Update bullets
    for i = #self.bullets, 1, -1 do
        local b = self.bullets[i]
        b.x = b.x + self.bullet_speed * math.cos(b.angle) * dt
        b.y = b.y + self.bullet_speed * math.sin(b.angle) * dt
        b.life = b.life - dt

        -- Remove old bullets
        if b.life < 0 then
            self.bullets[i] = self.bullets[#self.bullets]
            self.bullets[#self.bullets] = nil
        end
    end
end

function play_state:draw()
    -- Draw all units
    for name, unit in pairs(self.squad) do
        unit:draw()
    end

    -- Draw bullets
    love.graphics.setColor(1, 1, 0)
    for i, b in ipairs(self.bullets) do
        love.graphics.circle("fill", b.x, b.y, 3)
    end
end

function play_state:spawn_bullet(x, y, angle)
    table.insert(self.bullets, {x = x, y = y, angle = angle, life = self.bullet_lifetime})
end

function play_state:mousepressed(x, y, button)
    -- Player shoots on left click
    if button == 1 then
        self.squad["player"]:shoot_at(x, y)
    end
end

GameState.states["play"] = play_state

-- GameState Manager Functions
function GameState.switch(state_name, ...)
    if GameState.states[state_name] then
        if input and input.clear then
            input:clear()
        end
        GameState.current = GameState.states[state_name]
        if GameState.current.enter then
            GameState.current:enter(...)
        end
    end
end

function GameState.update(dt)
    if GameState.current.update then
        GameState.current:update(dt)
    end
end

function GameState.draw()
    if GameState.current.draw then
        GameState.current:draw()
    end
end

function GameState.mousepressed(x, y, button)
    if GameState.current.mousepressed then
        GameState.current:mousepressed(x, y, button)
    end
end

return GameState
