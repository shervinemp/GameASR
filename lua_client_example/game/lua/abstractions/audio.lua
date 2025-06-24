-- Audio Abstraction

Audio = {}
Audio.__index = Audio

function Audio.new()
  local audio = {
    sources = {},
    volume = 1.0
  }
  setmetatable(audio, Audio)
  return audio
end

function Audio:addSource(source)
  table.insert(self.sources, source)
end

function Audio:play(name)
  for _, src in ipairs(self.sources) do
    if src.name == name then
      src:play()
    end
  end
end

function Audio:setVolume(vol)
  self.volume = vol
  for _, src in ipairs(self.sources) do
    src:setVolume(vol)
  end
end

return Audio
