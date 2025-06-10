function draw_scene()
    love.graphics.clear(1, 1, 1)
    -- love.graphics.setBackgroundColor(0, 0, 0)
    love.graphics.draw(player_sprite, player.x, player.y, player.angle, 1, 1, 
                      player_sprite_width / 2, player_sprite_height / 2)
end

function love.load()

    game_states = {}
    current_state = "start"
    start_time = love.timer.getTime()

    -- Initialize player and window settings
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
    player_sprite = love.graphics.newImage("assets/player.png")
    player_sprite_width = player_sprite:getWidth()
    player_sprite_height = player_sprite:getHeight()

    -- Initialize game states
    game_states.start = {
        enter = function() print("Entering start state") end,
        update = function(dt) end,
        draw = function() love.graphics.print("Press any key to start", window.width / 2 - 100, window.height / 2) end,
        keypressed = function(key)
            if current_state == "start" then
                current_state = "play"
                game_states.play.enter()
            end
        end
    }
    game_states.play = {
        enter = function() print("Entering play state") end,
        update = function(dt) love.update(dt) end,
        draw = function() draw_scene() end,
        keypressed = function(key)
            if key == "escape" then
                current_state = "start"
                game_states.start.enter()
            elseif key == "p" then
                current_state = "over"
                game_states.over.enter()
            end
        end
    }
    game_states.over = {
        enter = function() print("Entering over state") end,
        update = function(dt) end,
        draw = function() love.graphics.print("Game Over! Press R to restart", window.width / 2 - 100, window.height / 2) end,
        keypressed = function(key)
            if key == "r" then
                current_state = "start"
                game_states.start.enter()
            end
        end
    }
end

function love.update(dt)
    if current_state == "play" then
        -- Handle input for play state
        y_dir = 1
        if love.keyboard.isDown("down") or love.keyboard.isDown("s") then
            player.x = player.x + player.speed * math.sin(player.angle) * dt
            player.y = player.y - player.speed * math.cos(player.angle) * dt
            y_dir = -1
        end
        if love.keyboard.isDown("up") or love.keyboard.isDown("w") then
            player.x = player.x - player.speed * math.sin(player.angle) * dt
            player.y = player.y + player.speed * math.cos(player.angle) * dt
            y_dir = 1
        end
        if love.keyboard.isDown("left") or love.keyboard.isDown("a") then
            player.angle = (player.angle - math.pi * dt * y_dir) % (2 * math.pi)
        end
        if love.keyboard.isDown("right") or love.keyboard.isDown("d") then
            player.angle = (player.angle + math.pi * dt * y_dir) % (2 * math.pi)
        end

        -- Handle shift for speed boost
        if love.keyboard.isDown("lshift") then
            player.speed = 200
        else
            player.speed = 100
        end

        -- Keep player within bounds
        player.x = math.max(player_sprite_width / 2, math.min(player.x, window.width - player_sprite_width / 2))
        player.y = math.max(player_sprite_height / 2, math.min(player.y, window.height - player_sprite_height / 2))
    end
end

function love.draw()
    game_states[current_state].draw()
end

function love.keypressed(key)
    game_states[current_state].keypressed(key)
end
