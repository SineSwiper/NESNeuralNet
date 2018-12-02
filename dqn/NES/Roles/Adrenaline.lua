-- Code for handling post-death savestate replays and adrenaline reward system
local Env = torch.getmetatable('NES.NesEnv')

local STATE_LEVEL_STEP = 10
local NUM_OF_RETRIES   = 5   -- retries per state

local inDeathSpiral    = false
local deathFrame       = 0
local curFrameInSpiral = 0
local retryCount       = 0

local prevStates = {
    [1]={},  -- 10^0 frame increments
}

function Env:saveState()
    -- Only save states outside of the death spiral
    if inDeathSpiral or self.romEnv:isGameOver() then return end

    local newState = savestate.object()
    savestate.save(newState)

    local cur = { state=newState, frame=emu.framecount() }

    -- Add to the list and shuffle around the old states
    table.insert(prevStates[1], cur)

    -- Check older states and see if they fit into the lower levels
    for i = 2, table.getn(prevStates)+1 do
        local lastLower      = table.getn(prevStates[i-1])
        if lastLower == 0 then break end

        local lowerState     = prevStates[i-1][1]  -- first state in lower (more granular) level
        local lowerIncrement = STATE_LEVEL_STEP ^ (i-2)
        local upperIncrement = STATE_LEVEL_STEP ^ (i-1)

        local lastUpper      = prevStates[i] and table.getn(prevStates[i]) or 0
        local frameDelta

        if lastUpper == 0 then
            -- No records at all: just check for the proper intervals from current
            frameDelta = emu.framecount() - lowerState.frame
        else
            local upperState = prevStates[i][lastUpper]  -- last state in upper (less granular) level
            frameDelta = lowerState.frame - upperState.frame
        end

        if frameDelta >= upperIncrement then
            if not prevStates[i] then prevStates[i] = {} end
            table.insert(prevStates[i], lowerState)
        end
    end

    -- Clean up older frames
    for i = 1, table.getn(prevStates) do
        if table.getn(prevStates[i]) > STATE_LEVEL_STEP then
            table.remove(prevStates[i], 1)
        end
    end
end

-- Checks the death spiral variables and returns an adrenaline value
function Env:checkDeathSpiral()
    local isDead = self.romEnv:isGameOver()

    local exitSpiralReward = 0
    if not inDeathSpiral and isDead then
        -- Entering a death spiral
        inDeathSpiral    = true
        deathFrame       = emu.framecount()
        curFrameInSpiral = deathFrame - self.romEnv:deathSpiralStartFrameDelta()
        retryCount       = 0
        print("Entering death spiral @ frame " .. deathFrame)
    elseif inDeathSpiral and not isDead and emu.framecount() >= deathFrame then
        -- Exiting a death spiral
        inDeathSpiral    = false
        deathFrame       = 0
        curFrameInSpiral = 0
        retryCount       = 0
        exitSpiralReward = 10000
        print("Exiting death spiral @ frame " .. emu.framecount())
    end

    -- TODO: Decaying exit reward

    if isDead then
        -- death penalty
        return -10000
    elseif not inDeathSpiral then
        -- not at risk (or just getting an adrenaline boost from success)
        return exitSpiralReward
    else
        -- quickly decaying adrenaline value
        local frameDelta = deathFrame - curFrameInSpiral
        return math.floor( 0.5 ^ (frameDelta-1) * 1000 )  -- ie: 1000, 500, 250, 125, etc.
    end
end

-- Tries to load a save state and returns terminal state
-- (inverse of whether it was successful or not)
function Env:loadState()
    -- Only load states inside of the death spiral
    if not inDeathSpiral then return end

    -- Increment retry count
    retryCount = retryCount + 1
    if retryCount > NUM_OF_RETRIES then
        retryCount = 1
        curFrameInSpiral = curFrameInSpiral - 1
    end

    -- Look for a state to load
    local cur
    for i = 1, table.getn(prevStates) do
        for j = table.getn(prevStates[i]), 1, -1 do
            if prevStates[i][j].frame <= curFrameInSpiral then
                cur = prevStates[i][j]
                print("Death spiral picked frame: ", cur.frame, i, j)
                break
            end
        end

        if cur then break end
    end

    -- Did we actually run out of save states??
    if not cur then
        return true
    end

    -- Reset the current frame to whatever we picked, in case it's lower than
    -- a single frame.  (Don't want to continue to pick it after retries...)
    curFrameInSpiral = cur.frame

    -- Make sure it doesn't disappear after load
    -- XXX: The retryCount check is hokey.  We should have a boolean to confirm
    -- that it is persistant.
    if retryCount == 1 then
        savestate.persist(cur.state)
    end

    -- Load it!
    savestate.load(cur.state)

    return false
end

function reloadFirstState()
    local lastIdx = table.getn(prevStates[1])
    if prevStates and lastIdx > 0 then
        savestate.load( prevStates[1][lastIdx] )
    end
end

function Env:clearAllStates()
    prevStates = {
        [1]={},
    }
    collectgarbage()  -- persistent save states use files

    -- Reset death spiral vars
    inDeathSpiral    = false
    deathFrame       = 0
    curFrameInSpiral = 0
    retryCount       = 0
end
