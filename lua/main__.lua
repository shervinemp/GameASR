local player = {
    x = 0,
    y = 0,
    angle = 0,
    speed = 100,
    width = 0,
    height = 0
}

local windowWidth, windowHeight = love.graphics.getWidth(), love.graphics.getHeight()

function setup()
    -- Initialize player sprite
    local img = love.graphics.newImage("player.png")
    player.width = img:getWidth()
    player.height = img:getHeight()
    
    -- Set initial position to center of screen
    player.x = windowWidth / 2 - player.width / 2
    player.y = windowHeight / 2 - player.height / 2
end

function drawScene()
    love.graphics.clear(1, 1, 1)
    love.graphics.translate(player.x, player.y)
    love.graphics.rotate(player.angle)
    
    -- Draw player sprite at origin point (center of image by default)
    love.graphics.draw(player Sprite, -player.width / 2, -player.height / 2, 0, 1, 1)
    
    -- Reset transformations
    love.graphics.pop()
end

function love.load()
    setup()
    love.window.setMode(1200, 800, {fullscreen = false, vsync = true})
end

function love.update(dt)
    local speedMultiplier = 1.0
    
    if love.keyboard.isDown("lshift") then
        speedMultiplier = 2.0
    end

    -- Handle movement
    if love.keyboard.isDown("w") or love.keyboard.isDown("up") then
        player.x = player.x - math.cos(player.angle) * player.speed * dt * speedMultiplier
    end
    if love.keyboard.isDown("s") or love.keyboard.isDown("down") then
        player.x = player.x + math.cos(player.angle) * player.speed * dt * speedMultiplier
    end
    if love.keyboard.isDown("a") or love.keyboard.isDown("left") then
        player.y = player.y - math.sin(player.angle) * player.speed * dt * speedMultiplier
    end
    if love.keyboard.isDown("d") or love.keyboard.isDown("right") then
        player.y = player.y + math.sin(player.angle) * player.speed * dt * speedMultiplier
    end

    -- Handle rotation
    if love.keyboard.isDown("left") or love.keyboard.isDown("a") then
        player.angle = (player.angle - 0.1 * dt) % (2 * math.pi)
    end
    if love.keyboard.isDown("right") or love.keyboard.isDown("d") then
        player.angle = (player.angle + 0.1 * dt) % (2 * math.pi)
    end

    -- Keep player within screen bounds
    local screenWidth, screenHeight = love.graphics.getWidth(), love.graphics.getHeight()
    player.x = math.max(player.width / 2, math.min(player.x, screenWidth - player.width / 2))
    player.y = math.max(player.height / 2, math.min(player.y, screenHeight - player.height / 2))
end

function love.draw()
    -- Set viewport to full screen
    love.graphics.setViewport(0, 0, windowWidth, windowHeight)
    
    drawScene()
end

function love.resize(w, h)
    windowWidth, windowHeight = w, h
    setup() -- Reset player position when screen size changes
end

function love.keypressed(key)
    if key == "escape" then
        love.event.quit()
    end
end
