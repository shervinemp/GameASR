function draw_scene()
    love.graphics.clear(1, 1, 1)
    -- love.graphics.setBackgroundColor(0, 0, 0)
    love.graphics.draw(player_sprite, player.x, player.y, player.angle, 1, 1, 
                      player_sprite_width / 2, player_sprite_height / 2)
end

function love.load()
    player = {
        x = 600,
        y = 400,
        angle = 0,
        speed = 100 -- Add speed here
    }
    window = {
        width = 1200,
        height = 800,
    }
    love.window.setMode(window.width, window.height)
    player_sprite = love.graphics.newImage("player.png")
    player_sprite_width = player_sprite:getWidth()
    player_sprite_height = player_sprite:getHeight()
end

function love.update(dt)
    if love.keyboard.isDown("up") or love.keyboard.isDown("w") then
        player.x = player.x - player.speed * math.sin(player.angle) * dt
        player.y = player.y + player.speed * math.cos(player.angle) * dt -- Fixed direction for up
    end
    if love.keyboard.isDown("down") or love.keyboard.isDown("s") then
        player.x = player.x + player.speed * math.sin(player.angle) * dt
        player.y = player.y - player.speed * math.cos(player.angle) * dt -- Fixed direction for down
    end
    if love.keyboard.isDown("left") or love.keyboard.isDown("a") then
        player.angle = (player.angle - math.pi * dt) % (2 * math.pi)
    end
    if love.keyboard.isDown("right") or love.keyboard.isDown("d") then
        player.angle = (player.angle + math.pi * dt) % (2 * math.pi)
    end
    if love.keyboard.isDown("lshift") then
        player.speed = 200
    else
        player.speed = 100
    end
    if love.keyboard.isDown("escape") then
        love.event.quit()
    end
    
    player.x = math.max(player_sprite_width / 2, math.min(player.x, window.width - player_sprite_width / 2))
    player.y = math.max(player_sprite_height / 2, math.min(player.y, window.height - player_sprite_height / 2))
end

function love.draw()
    draw_scene()
end