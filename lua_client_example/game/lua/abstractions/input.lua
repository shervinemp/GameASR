-- Input Abstraction

Input = {}
Input.__index = Input

function Input.new()
  local input = {
    keys = {},
    mouse = {x=0, y=0},
    active = false,
    pressedKeys = {}
  }
  setmetatable(input, Input)
  return input
end

function Input:registerKey(key, action)
  self.keys[key] = action
end

function Input:handleEvent(event)
  if event.type == "keypress" then
    local key = event.key
    -- Only call action if this is the first press (not a repeat)
    if not self.pressedKeys[key] and self.keys[key] then
      self.keys[key](event)
      self.pressedKeys[key] = true
    end
  elseif event.type == "keyrelease" then
    local key = event.key
    -- Reset the pressed flag on release
    self.pressedKeys[key] = false
  elseif event.type == "mouse_move" then
    self.mouse.x, self.mouse.y = event.x, event.y
  elseif event.type == "update" and event.dt then
    -- Do nothing for now, but this allows the physics update to proceed without recursive calls
  end
end

function Input:isKeyReleased(key)
  return not self.pressedKeys[key]
end

function Input:clear()
  self.pressedKeys = {}
end

return Input
