-- Physics Abstraction

Physics = {}
Physics.__index = Physics

function Physics.new()
  local physics = {
    bodies = {},
    gravity = {0, -9.81},
    collisionSystem = require("abstractions.collision").new()
  }
  setmetatable(physics, Physics)
  return physics
end

function Physics:addBody(body)
  table.insert(self.bodies, body)
  self.collisionSystem:addBody(body)
end

function Physics:applyMovement(body, dt)
  local dx = (body.dx or 0) * dt
  local dy = (body.dy or 0) * dt

  body.x = body.x + dx
  body.y = body.y + dy

  if not body.update then
    body.dx, body.dy = 0, 0
  end
end

function Physics:update(dt)
  for _, body in ipairs(self.bodies) do
    self:applyMovement(body, dt)

    if body.update then
      body:update(dt)  -- Pass only delta time to avoid recursion
    end
  end

  -- Update collision system after all bodies have moved
  self.collisionSystem:update(dt)
end

return Physics
