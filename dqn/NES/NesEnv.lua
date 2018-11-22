-- Standard NES specs
local RAM_LENGTH    = 2048
local SCREEN_WIDTH  = 256
local SCREEN_HEIGHT = 240

local FRAME2BUTTON = {
    ['R']='right',
    ['L']='left',
    ['D']='down',
    ['U']='up',
    ['T']='start',
    ['S']='select',
    ['B']='B',
    ['A']='A',
}
local BUTTON2FRAME = {}
local BUTTON_SET   = {}

for i=1,8,1 do
    local letter = string.sub('UDLRBAST', i, i)
    local button = FRAME2BUTTON[letter]
    table.insert(BUTTON_SET, button)
    BUTTON2FRAME[button] = letter
end

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
        -- Reward/penalty to give after dying
        gameOverReward=-10000,

        -- observation points
        useRGB=true,
        useRAM=false,
    }
    updateDefaults(self.config, extraConfig)

    require("NES.Games." .. GAME_NAME)
    self.romEnv = NES.RomEnv()

    -- Highest frame rate while still keeping all of the frames
    emu.speedmode('maximum')

    local obsShapes = {{SCREEN_HEIGHT, SCREEN_WIDTH}}
    if self.config.useRAM then
        obsShapes={{SCREEN_HEIGHT, SCREEN_WIDTH}, {RAM_LENGTH}}
    end
    self.envSpec = {
        nActions=self.romEnv:getNumLegalActions(),
        obsShapes=obsShapes,
    }

    self.trainingCache={}
    self:fillTrainingCache()
end

-- Returns a list of observations.
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
        return self:_generateObservations(), self.config.gameOverReward, true  -- true = terminal
    end

    local btns = self:_action2btn(action)
    self:act(btns)
    self:_displayButtons(btns)

    local reward = self.romEnv:reward()
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
            local letter = BUTTON2FRAME[btn]
            local color = 'white'
            if pBtns[btn] == true then color = 'red' end

            local x = _ * 7 + xOff
            local y = (player-1) * 10 + yOff

            gui.text(x, y, letter, color, 'black')
        end
    end
end

function Env:_createRGBObs()
    -- Grab an entire screenshot as a string (then remove the GD header)
    local screen_str = gui.gdscreenshot(true)
    screen_str = string.sub(screen_str, 12, -1)

    -- Add the screen string into a 3D, ARGB ByteTensor
    local screen_tensor = torch.ByteTensor(SCREEN_HEIGHT, SCREEN_WIDTH, 4)
    screen_tensor:storage():string(screen_str)

    -- Unfortunately, the color channels are in the wrong dimension, so we need to move
    -- them around.  While we're here, remove the Alpha channel.
    local byte_obs = torch.ByteTensor(3, SCREEN_HEIGHT, SCREEN_WIDTH)
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
    local ram_str = memory.readbyterange(0, RAM_LENGTH)

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

function Env:fillTrainingCache()
    -- XXX: Hard-coded directory here...
    local dir = '../movies/training'

    -- Find movie files to use and sort them
    local fm_files = {}
    for file in lfs.dir(dir) do
        local full_path = dir .. '/' .. file
        if string.match(file, '.fm2$') and lfs.attributes(full_path, 'mode') == 'file' then
            table.insert(fm_files, full_path)
        end
    end
    table.sort(fm_files)

    -- Read the movie files
    for _, full_path in ipairs(fm_files) do
        local skip_frames = self.romEnv:getNumSkipFrames()

        for line in io.lines(full_path) do
            --- XXX: Need header sanity checks here
            
            if string.match(line, '^|%d+|.*|.*|.*|.*$') then
                if skip_frames > 0 then
                    skip_frames = skip_frames - 1
                    -- Insert reset line
                    if skip_frames == 0 then table.insert(self.trainingCache, '|1||||') end
                else
                    -- Frame data; process this later
                    table.insert(self.trainingCache, line)
                end
            end
        end

        print("Loaded " .. full_path .. " into training cache")
    end
end

function Env:fillTrainingActions(actions, step, min, len)
    -- Random bits
    if not len then len = torch.random(1, 300) end  -- upwards of 5 seconds
    if not min then min = torch.random(table.getn(self.trainingCache) - len) end
    local max = min + len - 1

    -- Button table prep
    local btnSet = self.romEnv:getLegalButtonSet()
    local idxSet = {}  -- reverse lookup
    for _, btn in pairs(btnSet) do
        local key = btn[1] .. ',' .. BUTTON2FRAME[btn[2]]
        idxSet[key] = _
    end 

    -- Process frame lines (with dynamic loop control)
    local i = min
    while i <= max do
        --- XXX: Only supports Player 1 right now
        local movie_line = self.trainingCache[i]
        local flags, player_one = string.match(movie_line, '^|(%d+)|([^|]*)|')

        -- Special action to reset the game.  Only used if it appears that
        -- full human training is requested.  Note that in normal *.fm2 files
        -- the rest of the line applies (but is usually blank), but not in the
        -- manner we're using it above (with "|1||||").
        if flags == '1' then
            if step == 1 and min == 1 then table.insert(actions, { 'reset' }) end
            i = i + 1

        -- Early stages of learning needs more active development, so skip
        -- blank actions when the step isn't very accurate, anyway.
        elseif (step >= 4 and not string.match(player_one, '[A-Z]')) then
            i = i + 1
            max = math.min(max + 1, table.getn(self.trainingCache))

        -- Standard action
        else
            local action = {}
            for j=1,string.len(player_one),1 do
                local b = string.sub(player_one, j, j)
                if b ~= '.' and FRAME2BUTTON[b] then
                    local key = '1,' .. b
                    local val = idxSet[key]
                    if val then table.insert(action, val) end
                end
            end

            table.insert(actions, action)
            i = i + step
        end
    end
end

function Env:resetGame()
    emu.poweron()
    self.romEnv:skipStartScreen()
end
