Collision = {}

function Collision.new()
  local self = {
    bodies = {}
  }
  setmetatable(self, {__index=Collision})
  return self
end

-- Adds a body to the collision system
function Collision:addBody(body)
  table.insert(self.bodies, body)
end

-- Removes a body from the collision system
function Collision:removeBody(body)
  for i, b in ipairs(self.bodies) do
    if b == body then
      table.remove(self.bodies, i)
      break
    end
  end
end

-- Checks if two bodies collide using axis-aligned bounding box (AABB) collision detection
function Collision:checkCollision(body1, body2)
  return not (
    body1.x + body1.width / 2 < body2.x - body2.width / 2 or
    body1.x - body1.width / 2 > body2.x + body2.width / 2 or
    body1.y + body1.height / 2 < body2.y - body2.height / 2 or
    body1.y - body1.height / 2 > body2.y + body2.height / 2
  )
end

-- Checks for collisions between a body and all other bodies in the system
function Collision:checkAllCollisions(body)
  local collisions = {}
  for _, other in ipairs(self.bodies) do
    if self:checkCollision(body, other) then
      table.insert(collisions, other)
    end
  end
  return collisions
end

-- Updates all bodies and handles collision response (bounce back)
function Collision:update(dt)
  local displacements = {}
  for i = 1, #self.bodies do
    displacements[i] = {x = 0, y = 0}
  end

  for i = 1, #self.bodies do
    local body1 = self.bodies[i]
    for j = i + 1, #self.bodies do
      local body2 = self.bodies[j]
      if self:checkCollision(body1, body2) then
        -- Simple collision response - bounce back bodies
        local overlapX = math.max((body1.x + body1.width / 2 - (body2.x - body2.width / 2)), (body2.x + body2.width / 2 - (body1.x - body1.width / 2)))
        local overlapY = math.max((body1.y + body1.height / 2 - (body2.y - body2.height / 2)), (body2.y + body2.height / 2 - (body1.y - body1.height / 2)))

        if overlapX < overlapY then
          -- X-axis collision
          if body1.x < body2.x then
            displacements[i].x = displacements[i].x - overlapX * 0.5
            displacements[j].x = displacements[j].x + overlapX * 0.5
          else
            displacements[i].x = displacements[i].x + overlapX * 0.5
            displacements[j].x = displacements[j].x - overlapX * 0.5
          end

          -- Reverse X velocity
          local tempVx = body1.dx or 0
          body1.dx = (body2.dx or 0)
          body2.dx = tempVx
        else
          -- Y-axis collision
          if body1.y < body2.y then
            displacements[i].y = displacements[i].y - overlapY * 0.5
            displacements[j].y = displacements[j].y + overlapY * 0.5
          else
            displacements[i].y = displacements[i].y + overlapY * 0.5
            displacements[j].y = displacements[j].y - overlapY * 0.5
          end

          -- Reverse Y velocity
          local tempVy = body1.dy or 0
          body1.dy = (body2.dy or 0)
          body2.dy = tempVy
        end
      end
    end
  end

  -- Apply accumulated displacements
  for i = 1, #self.bodies do
    local body = self.bodies[i]
    if displacements[i].x ~= 0 or displacements[i].y ~= 0 then
      body.x = body.x + displacements[i].x
      body.y = body.y + displacements[i].y
    end
  end
end

-- Checks if a body is within the screen bounds
function Collision:isWithinBounds(body, windowWidth, windowHeight)
  local sprite_width = body.sprite:getWidth()
  local sprite_height = body.sprite:getHeight()

  return (
    body.x + sprite_width / 2 >= 0 and
    body.y + sprite_height / 2 >= 0 and
    body.x - sprite_width / 2 <= windowWidth and
    body.y - sprite_height / 2 <= windowHeight
  )
end

return Collision
