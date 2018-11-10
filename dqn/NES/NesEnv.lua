-- Standard NES specs
local RAM_LENGTH    = 2048
local SCREEN_WIDTH  = 256
local SCREEN_HEIGHT = 240

-- TODO: Move to main config parameters
-- Configurable game name
local GAME_NAME = 'SuperMarioBros'

-- Copies values from src to dst.
local function update(dst, src)
    for k, v in pairs(src) do
        dst[k] = v
    end
end

-- Copies the config. An error is raised on unknown params.
local function updateDefaults(dst, src)
    for k, v in pairs(src) do
        if dst[k] == nil then
            _print_usage(dst)
            error("unsupported param: " .. k)
        end
    end
    update(dst, src)
end

local Env = torch.class('NES.NesEnv')
function Env:__init(extraConfig)
    self.config = {
        -- An additional reward signal can be provided
        -- after the end of one game.
        -- Note that many games don't change the score
        -- when loosing or gaining a life.
        gameOverReward=0,
        -- Screen display can be enabled.
        display=false,
        -- The RAM can be returned as an additional observation.
        enableRamObs=false,
    }
    updateDefaults(self.config, extraConfig)

    require("NES.Games." .. GAME_NAME)
    self.romEnv = NES.RomEnv()

    emu.speedmode('maximum')

    local obsShapes = {{SCREEN_HEIGHT, SCREEN_WIDTH}}
    if self.config.enableRamObs then
        obsShapes={{SCREEN_HEIGHT, SCREEN_WIDTH}, {RAM_LENGTH}}
    end
    self.envSpec = {
        nActions=self.romEnv:getNumLegalActions(),
        obsShapes=obsShapes,
    }
end

-- Returns a description of the observation shapes
-- and of the possible actions.
function Env:getEnvSpec()
    return self.envSpec
end

-- Returns a list of observations.
-- The integer palette values are returned as the observation.
function Env:envStart()
    self:resetGame()
    return self:_generateObservations()
end

-- Does the specified actions and returns the (reward, observations) pair.
-- Valid actions:
--     {torch.Tensor(zeroBasedAction)}
function Env:envStep(actions)
    assert(#actions == 1, "one action is expected")
    assert(actions[1]:nElement() == 1, "one discrete action is expected")

    if self.romEnv:isGameOver() then
        self:resetGame()
        -- The first screen of the game will be also
        -- provided as the observation.
        return self.config.gameOverReward, self:_generateObservations()
    end

    local reward = self.romEnv:act(actions[1][1])
    return reward, self:_generateObservations()
end

function Env:_createObs()
    local obs = torch.FloatTensor(3, SCREEN_HEIGHT, SCREEN_WIDTH)
    local obs_data = torch.data(obs)

    for y=0,SCREEN_HEIGHT-1 do
        for x=0,SCREEN_WIDTH-1 do
            local pixel_index = (y*SCREEN_WIDTH + x) * 3 + 1

            local r,g,b,palette = emu.getscreenpixel(x, y, true)
            obs_data[pixel_index + 0] = r / 255
            obs_data[pixel_index + 1] = g / 255
            obs_data[pixel_index + 2] = b / 255
        end
    end

    return obs
end

function Env:_createRamObs()
    local ram = torch.ByteTensor(RAM_LENGTH)
    local ram_data = torch.data(ram)

    for i=0,RAM_LENGTH-1,1 do
        ram_data[i] = memory.readbyte(i)
    end

    return ram
end

-- Generates the observations for the current step.
function Env:_generateObservations()
    local obs = self:_createObs()

    if self.config.enableRamObs then
        local ram = self:_createRamObs()
        return {obs, ram}
    else
        return {obs}
    end
end

function Env:buildActionsTensor()
    local btn = torch.ByteTensor(self.envSpec.nActions)
    local btn_data = torch.data(btn)

    for i=0,self.envSpec.nActions-1,1 do
        btn_data[i] = i
    end

    return btn
end

function Env:resetGame()
    emu.softreset()
    self.romEnv:skipStartScreen()
end

function Env:saveState()
    -- NOT IMPLEMENTED!
end

function Env:loadState()
    -- NOT IMPLEMENTED!
end

function Env:saveSnapshot()
    -- NOT IMPLEMENTED!
end

function Env:restoreSnapshot(snapshot)
    -- NOT IMPLEMENTED!
end
