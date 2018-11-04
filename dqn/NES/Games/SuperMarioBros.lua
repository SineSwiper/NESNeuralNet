-- TODO: Adapt CNN to use multiple dimensions for output to support multiple buttons

local LEGAL_ACTIONS = {
    [0]  = {},
    [1]  = { A=true     },
    [2]  = { B=true     },
    [3]  = { up=true    },
    [4]  = { right=true },
    [5]  = { left=true  },
    [6]  = { down=true  },
    [7]  = { left=true, right=true },
    [8]  = { A=true, up=true    },
    [9]  = { A=true, right=true },
    [10] = { A=true, left=true  },
    [11] = { A=true, down=true  },
    [12] = { A=true, left=true, right=true },
    [13] = { B=true, up=true    },
    [14] = { B=true, right=true },
    [15] = { B=true, left=true  },
    [16] = { B=true, down=true  },
    [17] = { B=true, left=true, right=true },
    [18] = { A=true, B=true, up=true    },
    [19] = { A=true, B=true, right=true },
    [20] = { A=true, B=true, left=true  },
    [21] = { A=true, B=true, down=true  },
    [22] = { A=true, B=true, left=true, right=true },
}

-- Fill in the rest of the buttons with false
for _, btn_table in pairs(LEGAL_ACTIONS) do
    for btn='up','down','left','right','A','B','start','select' do
        if btn_table[btn] == nil then btn_table[btn] = false end
    done
done

local RomEnv = torch.class('NES.RomEnv')
function RomEnv:__init()
    self.prevVals = {}
end

function RomEnv:skipStartScreen()
    -- Run a few frames first to get to the startup screen.
    for i=1,60,1 do
        emu.frameadvance()
    end
    -- Hit the start button
    for i=1,10,1 do
        joypad.set(1, { start=true })
        emu.frameadvance()
    end
end

-- FIXME: Remove lives()

function RomEnv:getCurrentScore()
    local score = 0

    score += memory.readbyteunsigned(0x07dd) * 1000000
    score += memory.readbyteunsigned(0x07de) *  100000
    score += memory.readbyteunsigned(0x07df) *   10000
    score += memory.readbyteunsigned(0x07e0) *    1000
    score += memory.readbyteunsigned(0x07e1) *     100
    score += memory.readbyteunsigned(0x07e2) *      10

    return score
end

function RomEnv:isGameOver()
    -- There are several different death indicators.  Check them all to
    -- exit as fast as possible.

    -- [0x000E] PlayerState: 0x06 Player dies, 0x0B Dying
    local player_state = memory.readbyteunsigned(0x000E)
    if player_state == 6 || player_state == 11 then return true end

    -- [0x0712] DeathMusicLoaded: boolean, usually used for falling deaths
    local death_music_loaded = memory.readbyteunsigned(0x0712)
    if death_music_loaded > 0 then return true end

    -- [0x0770] GameState: 02 = End game (dead)
    local game_state = memory.readbyteunsigned(0x0770)
    if game_state == 2 then return true end

    return false
end

function RomEnv:getLegalActionSet()
    return LEGAL_ACTIONS
end

function RomEnv:getNumLegalActions()
    return #LEGAL_ACTIONS + 1
end

-- Applies an action to the game and returns the reward. It is the user's responsibility
-- to check if the game has ended and reset when necessary - this method will keep pressing
-- buttons on the game over screen.
function RomEnv:act(act_num)
    local action = LEGAL_ACTIONS[act_num]
    assert(type(action) == 'table', "one action is expected")

    -- Set the action
    joypad.set(1, action)

    -- Frame advance
    emu.frameadvance()

    -- Reward computation
    local reward = 0

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
