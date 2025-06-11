-- Rendering Abstraction

Rendering = {}
Rendering.__index = Rendering

function Rendering.new()
  local rendering = {
    elements = {},
    visible = true
  }
  setmetatable(rendering, Rendering)
  return rendering
end

function Rendering:addElement(element)
  table.insert(self.elements, element)
end

function Rendering:draw()
  if not self.visible then return end
  for _, elem in ipairs(self.elements) do
    if elem.draw then
      elem:draw()
    end
  end
end

return Rendering
