-- UI Abstraction

UI = {}
UI.__index = UI

function UI.new()
  local ui = {
    elements = {},
    visible = false
  }
  setmetatable(ui, UI)
  return ui
end

function UI:addElement(element)
  table.insert(self.elements, element)
end

function UI:show()
  self.visible = true
end

function UI:hide()
  self.visible = false
end

function UI:update(dt)
  for _, elem in ipairs(self.elements) do
    if elem.update then
      elem:update(dt)
    end
  end
end

function UI:draw()
  if not self.visible then return end
  for _, elem in ipairs(self.elements) do
    if elem.draw then
      elem:draw()
    end
  end
end

return UI
