-- TODO: Merge and refactor

--[[
Game class that provides an interface for the roms

In general, you would want to use:
    NES.game(gamename)
]]

require 'torch'
local game = torch.class('NES.game')

--[[
Parameters:

 * `gamename` (string) - one of the rom names without '.zip' extension.
 * `options`  (table) - a table of options

Where `options` has the following keys:

 * `useRGB`   (bool) - true if you want to use RGB pixels.
 * `useRAM`   (bool) - true if you want to use the RAM.

]]
function game:__init(gamename, options)
    options = options or {}

    self.useRGB = options.useRGB
    self.useRAM = options.useRAM

    self.name         = gamename
    self.env          = NES.NesEnv({})
    self.observations = self.env:envStart()
    self.action       = {torch.Tensor{0}}

    self.game_over = function() return self.env.nes:isGameOver() end

    -- setup initial observations by playing a no-action command
    self:saveState()
    local x = self:play(0)
    self.observations[1] = x.data
    self:loadState()
end

-- FIXME: Remove all of this nonsense

function game:stochastic()
    return false
end


function game:shape()
    return self.observations[1]:size():totable()
end


function game:nObsFeature()
    return torch.prod(torch.Tensor(self:shape()),1)[1]
end


function game:saveState()
    self.env:saveState()
end


function game:loadState()
    return self.env:loadState()
end

function game:actions()
    return self.env:buildActionsTensor():storage():totable()
end

-- Ugly chain of function calls :(
function game:resetGame()
    self.env:resetGame()
end

function game:getCurrentScore()
    return self.env.nes:getCurrentScore()
end


--[[
Parameters:
 * `action` (int), the action to play

Returns a table containing the result of playing given action, with the
following keys:
 * `reward` - reward obtained
 * `data`   - observations
 * `pixels` - pixel space observations
 * `ram`    - ram of the ATARI if requested
 * `terminal` - (bool), true if the new state is a terminal state
]]
function game:play(action)
    action = action or 0
    self.action[1][1] = action

    -- take the step in the environment
    local reward, observations = self.env:envStep(self.action)
    local is_game_over = self.game_over(reward)

    local pixels = observations[1]
    local ram    = observations[2]
    local data   = pixels
    local gray   = pixels

    return {reward=reward, data=data, pixels=pixels, ram=ram,
            terminal=is_game_over, gray=gray}
end


function game:getState()
    return self.env:saveSnapshot()
end


function game:restoreState(state)
    self.env:restoreSnapshot(state)
end
