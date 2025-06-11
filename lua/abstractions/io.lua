-- IO Abstraction

IO = {}
IO.__index = IO

function IO.new()
  local io = {
    handlers = {},
    active = false
  }
  setmetatable(io, IO)
  return io
end

function IO:registerHandler(event, handler)
  self.handlers[event] = handler
end

function IO:dispatchEvent(event, data)
  if self.handlers[event] then
    self.handlers[event](data)
  end
end

return IO
