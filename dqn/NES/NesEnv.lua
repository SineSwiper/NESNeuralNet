-- Standard NES specs
local RAM_LENGTH    = 2048
local SCREEN_WIDTH  = 256
local SCREEN_HEIGHT = 240

local BUTTON_SET = {'up','down','left','right','A','B','start','select'}

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
function Env:envStep(action)
    assert(action, 'action is required')
    assert(type(action) == 'table', 'action needs to be a table')

    if self.romEnv:isGameOver() then
        self:resetGame()
        -- The first screen of the game will be also
        -- provided as the observation.
        return self.config.gameOverReward, self:_generateObservations()
    end

    local btns = self:_action2btn(action)
    self:act(btns)
    self:_displayButtons(btns)

    local reward = self.romEnv:reward()
    return reward, self:_generateObservations()
end

function Env:_action2btn(action)
    local btns   = {}
    local btnSet = self.romEnv:getLegalButtonSet()

    -- Fill in the true values
    for _, btnIsOn in pairs(action) do
        local btn    = btnSet[_]
        local player = btn[1]
        local bName  = btn[2]

        local btnBool = false
        if btnIsOn > 0 then btnBool = true end

        if btns[player] == nil then btns[player] = {} end
        btns[player][bName] = btnBool
    end

    -- Fill in the rest of the buttons with false
    for _, pBtns in pairs(btns) do
        for _, btn in pairs(BUTTON_SET) do
            if pBtns[btn] == nil then pBtns[btn] = false end
        end
    end

    return btns
end

-- Applies an action to the game
function Env:act(btns)
    assert(type(btns) == 'table', "buttons should be a table")

    -- Set the action
    for _, pBtns in pairs(btns) do
        joypad.set(_, pBtns)
    end

    -- Frame advance
    emu.frameadvance()
end

function Env:_displayButtons(btns)
    local xOff, yOff = self.romEnv:btnDisplayPos()

    for player, pBtns in pairs(btns) do
        for _, btn in pairs(BUTTON_SET) do
            local letter = string.upper( string.sub(btn, 1, 1) )
            local color = 'white'
            if pBtns[btn] == true then color = 'red' end

            local x = _ * 7 + xOff
            local y = (player-1) * 10 + yOff

            gui.text(x, y, letter, color, 'black')
        end
    end
end

function Env:_createObs()
    -- Grab an entire screenshot as a string (then remove the GD header)
    local screen_str = gui.gdscreenshot()
    screen_str = string.sub(screen_str, 12, -1)

    local RGBA_storage = torch.ByteStorage():string(screen_str)

    -- Add the storage into a 3D, RGBA ByteTensor
    -- (storage, storageOffset, sz1, st1 ... )
    local screen_tensor = torch.ByteTensor(RGBA_storage, 1, 4, 0, SCREEN_HEIGHT, 0, SCREEN_WIDTH, 0)

    -- Slice out the Alpha dimension and convert to Float
    local obs = screen_tensor:narrow(1, 2, 3):type('torch.FloatTensor')
    obs:div(255)

    return obs
end

function Env:_createRamObs()
    -- Grab all of the RAM (as a string)
    local ram_str = memory.readbyterange(0, RAM_LENGTH)
    local ram_storage = torch.ByteStorage():string(ram_str)

    -- Add the storage into a Byte Tensor
    -- (storage, storageOffset, sz1, st1)
    local obs = torch.ByteTensor(ram_storage, 1, RAM_LENGTH, 0)

    return obs
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
