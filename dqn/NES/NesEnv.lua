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
            error("unsupported param: " .. k)
        end
    end
    update(dst, src)
end

local Env = torch.class('NES.NesEnv')
require 'NES.Roles.Constants'
require 'NES.Roles.Displays'
require 'NES.Roles.Training'
require 'NES.Roles.Adrenaline'

function Env:__init(extraConfig)
    self.config = {
        -- The ROM name being played
        gamename='',

        -- Reward/penalty to give after dying
        gameOverReward=-10000,

        -- observation points
        useRGB=true,
        useRAM=false,
    }
    updateDefaults(self.config, extraConfig)

    require("NES.Games." .. self.config.gamename)
    self.romEnv = NES.RomEnv()

    -- Highest frame rate while still keeping all of the frames
    emu.speedmode('maximum')

    local obsShapes = {{NES.SCREEN_HEIGHT, NES.SCREEN_WIDTH}}
    if self.config.useRAM then
        obsShapes={{NES.SCREEN_HEIGHT, NES.SCREEN_WIDTH}, {NES.RAM_LENGTH}}
    end
    self.envSpec = {
        nActions=self.romEnv:getNumLegalActions(),
        obsShapes=obsShapes,
    }

    self.trainingCache={}
    self:fillTrainingCache()
end

-- Returns a list of observations.
function Env:envStart(movie_filename)
    self:resetGame(movie_filename)
    return self:_generateObservations()
end

function Env:resetGame(movie_filename)
    -- If recording a movie, stop here
    if movie.active() then movie.stop() end

    self:clearAllStates()

    if movie_filename then
        -- This will powercycle the NES for us
        local file = movie_filename:match("^.+/(.+)$")
        print("Recording " .. file)
        movie.record(movie_filename, 0, 'NES Neural Net AI')
    else
        emu.poweron()
    end

    self.romEnv:skipStartScreen()
end

-- Does the specified actions and returns the (reward, observations) pair.
function Env:envStep(action)
    assert(action, 'action is required')
    assert(type(action) == 'table', 'action needs to be a table')

    -- Handle adrenaline and save states
    local adrenaline = self:checkDeathSpiral()

    -- What are we doing with the save states?
    local terminal = false
    if self.romEnv:isGameOver() then
        terminal = self:loadState()
    else
        self:saveState()
    end

    -- Rare, but we might have ran out of save states...
    if terminal then
        -- Go back to the latest frame state, just for movie playback
        self:reloadFirstState()

        -- The first screen of the game will be also provided as the observation.
        return self:_generateObservations(), self.config.gameOverReward, terminal
    end

    local btns = self:_action2btn(action)
    self:act(btns)
    self:displayButtons(btns)

    local reward = self.romEnv:reward() + adrenaline
    return self:_generateObservations(), reward, false
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
        for _, btn in pairs(NES.BUTTON_SET) do
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

function Env:_createRGBObs()
    -- Grab an entire screenshot as a string (then remove the GD header)
    local screen_str = gui.gdscreenshot(true)
    screen_str = screen_str:sub(12, -1)

    -- Add the screen string into a 3D, ARGB ByteTensor
    local screen_tensor = torch.ByteTensor(NES.SCREEN_HEIGHT, NES.SCREEN_WIDTH, 4)
    screen_tensor:storage():string(screen_str)

    -- Unfortunately, the color channels are in the wrong dimension, so we need to move
    -- them around.  While we're here, remove the Alpha channel.
    local byte_obs = torch.ByteTensor(3, NES.SCREEN_HEIGHT, NES.SCREEN_WIDTH)
    for i=1,3,1 do
        byte_obs[i] = screen_tensor:select(3, i+1)
    end

    -- Convert to float
    local obs = byte_obs:type('torch.FloatTensor')
    obs:div(255)

    return obs
end

function Env:_createRAMObs()
    -- Grab all of the RAM (as a string)
    local ram_str = memory.readbyterange(0, NES.RAM_LENGTH)

    -- Add the storage into a Byte Tensor
    local obs = torch.ByteTensor(RAM_LENGTH)
    obs:storage():string(ram_str)

    return obs
end

-- Generates the observations for the current step.
function Env:_generateObservations()
    local obs_table = {}

    if self.config.useRGB then
        obs_table[1] = self:_createRGBObs()
    end
    if self.config.useRAM then
        obs_table[2] = self:_createRAMObs()
    end

    return obs_table
end
