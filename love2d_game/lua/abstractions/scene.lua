-- Scene Abstraction

Scene = {}
Scene.__index = Scene

function Scene.new()
  local scene = {
    objects = {},
    active = false
  }
  setmetatable(scene, Scene)
  return scene
end

function Scene:addObject(object)
  table.insert(self.objects, object)
end

function Scene:update(dt)
  for _, obj in ipairs(self.objects) do
    if obj.update then
      obj:update(dt)
    end
  end
end

function Scene:draw()
  for _, obj in ipairs(self.objects) do
    if obj.draw then
      obj:draw()
    end
  end
end

return Scene
