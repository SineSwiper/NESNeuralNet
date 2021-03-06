-- Game-specific functions for Super Mario Bros

-- Player # then button
local LEGAL_BUTTONS = {{1,'up'},{1,'down'},{1,'left'},{1,'right'},{1,'A'},{1,'B'}}

local RomEnv = torch.class('NES.RomEnv')
function RomEnv:__init()
    self.prevVals = {
        score = 0,
    }
end

function RomEnv:skipStartScreen()
    -- Run a few frames first to get to the startup screen.
    for i=1,33,1 do
        emu.frameadvance()
    end

    -- Hit the start button
    joypad.set(1, { start=true })
    emu.frameadvance()

    -- Skip the 1-1 screen
    for i=1,162,1 do
        emu.frameadvance()
    end
end

function RomEnv:getNumSkipFrames()
    return 33+1+162
end

function RomEnv:getCurrentScore()
    -- [0x07DD-0x07E2] Mario score in BCD format
    local score = 0

    score = score + memory.readbyteunsigned(0x07dd) * 1000000
    score = score + memory.readbyteunsigned(0x07de) *  100000
    score = score + memory.readbyteunsigned(0x07df) *   10000
    score = score + memory.readbyteunsigned(0x07e0) *    1000
    score = score + memory.readbyteunsigned(0x07e1) *     100
    score = score + memory.readbyteunsigned(0x07e2) *      10

    return score
end

function RomEnv:isGameOver()
    -- There are several different death indicators.  Check them all to
    -- exit as fast as possible.

    -- [0x000E] PlayerState: 0x06 Player dies, 0x0B Dying
    local player_state = memory.readbyteunsigned(0x000E)
    if player_state == 6 or player_state == 11 then return true end

    -- [0x0712] DeathMusicLoaded: boolean, usually used for falling deaths
    local death_music_loaded = memory.readbyteunsigned(0x0712)
    if death_music_loaded > 0 then return true end

    -- [0x0770] GameState: 02 = End game
    -- XXX: Can't trust this.  End game also happens during world changes.

    return false
end

function RomEnv:getLegalButtonSet()
    return LEGAL_BUTTONS
end

function RomEnv:getNumLegalActions()
    return #LEGAL_BUTTONS
end

function RomEnv:btnDisplayPos()
    return 0, 35
end

function RomEnv:heatmapDisplayPos()
    return 256-84-1, 35
end

-- Returns the reward, based on what's in the RAM after the action.
function RomEnv:reward()
    -- Sitting still is not a good look, so the default score is negative
    local reward = -10

    -- [0x0057] Player horizontal speed
    local x_speed = memory.readbytesigned(0x0057)

    -- More reward for going fast to the right.  Constant reward; no need for delta.
    reward = reward + x_speed * 5

    -- Add in a score delta
    local new_score = self:getCurrentScore()
    if self.prevVals['score'] > 0 and new_score > 0 then
        reward = reward + new_score - self.prevVals['score']
    end
    self.prevVals['score'] = new_score

    return reward
end

function RomEnv:deathSpiralStartFrameDelta()
    -- [0x07F8/A] Game Timer in BCD Format
    local timer = 0
    timer = timer + memory.readbytesigned(0x07f8) * 100
    timer = timer + memory.readbytesigned(0x07f9) *  10
    timer = timer + memory.readbytesigned(0x07fa) *   1

    if timer < 10 then
        -- Taking too long: We demand perfection, so start the whole level over again
        return (400 - timer) * 24  -- 400s timer * frames/sec
    end

    return 1
end