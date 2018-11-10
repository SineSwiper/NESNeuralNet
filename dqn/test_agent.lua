--[[
Copyright (c) 2014 Google Inc.
See LICENSE file for full terms of limited license.
]]

gd = require "gd"

require "options.test"

if not dqn then
    require "initenv"
end

local opt = globalDQNOptions

--- General setup.
local game_env, game_actions, agent, opt = setup(opt)

-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

-- file names from command line
local gif_filename = opt.gif_file

-- start a new game
local screen, reward, terminal = game_env:newGame()

-- compress screen to JPEG with 100% quality
local jpg = image.compressJPG(screen:squeeze(), 100)
-- create gd image from JPEG string
local im = gd.createFromJpegStr(jpg:storage():string())
-- convert truecolor to palette
im:trueColorToPalette(false, 256)

-- write GIF header, use global palette and infinite looping
im:gifAnimBegin(gif_filename, true, 0)
-- write first frame
im:gifAnimAdd(gif_filename, false, 0, 0, 7, gd.DISPOSAL_NONE)

-- remember the image and show it first
local previm = im
local win = nil
-- local win = image.display({image=screen})

print("Started playing...")

-- play one episode (game)
while not terminal do
    -- if action was chosen randomly, Q-value is 0
    agent.bestq = 0
    
    -- choose the best action
    local action_index = agent:perceive(reward, screen, terminal, true, 0.05)

    -- play game in test mode (episodes don't end when losing a life)
    screen, reward, terminal = game_env:step(game_actions[action_index], false)

    -- display screen
    -- image.display({image=screen, win=win})

    -- create gd image from tensor
    jpg = image.compressJPG(screen:squeeze(), 100)
    im = gd.createFromJpegStr(jpg:storage():string())
    
    -- use palette from previous (first) image
    im:trueColorToPalette(false, 256)
    im:paletteCopy(previm)

    -- write new GIF frame, no local palette, starting from left-top, 7ms delay
    im:gifAnimAdd(gif_filename, false, 0, 0, 7, gd.DISPOSAL_NONE)
    -- remember previous screen for optimal compression
    previm = im

end

-- end GIF animation and close CSV file
gd.gifAnimEnd(gif_filename)

print("Finished playing, close window to exit!")
