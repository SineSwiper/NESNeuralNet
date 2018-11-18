-- TODO: Refactor and merge

-- This file defines the NES.GameEnvironment class.

-- The GameEnvironment class.
local gameEnv = torch.class('NES.GameEnvironment')

function gameEnv:__init(_opt)
    local _opt = _opt or {}

    self.verbose = _opt.verbose or 0
    self._actrep = _opt.actrep or 1
    self._state  = {}

    return self
end

function gameEnv:_updateState(data, reward, terminal)
    self._state.observation  = data
    self._state.reward       = reward
    self._state.terminal     = terminal
    return self
end


function gameEnv:getState()
    return self._state.observation, self._state.reward, self._state.terminal
end

function gameEnv:reset(_env, _params)
    -- TODO: Use emu.romname
    local game_name = _env
    local params = _params or {useRGB=true}

    self.NesEnv = NES.NesEnv(params)

    -- start the game
    if self.verbose > 0 then
        print('\nPlaying:', game_name)
    end
    self:newGame()
    return self
end

-- Function plays `action` in the game and return game state.
function gameEnv:step(action, training)
    -- Convert ByteTensor to table
    if torch.isTensor(action) then
        action = action:totable()
    end

    -- accumulate rewards over actrep action repeats
    local cumulated_reward = 0
    local data, reward, terminal
    for i=1,self._actrep do
        -- Take selected action
        data, reward, terminal = self.NesEnv:envStep(action)
        data = data[1]  -- pixels, will get to RAM later on...

        -- accumulate instantaneous reward
        cumulated_reward = cumulated_reward + reward

        -- game over, no point to repeat current action
        if terminal then break end
    end
    self:_updateState(data, cumulated_reward, terminal)
    return self:getState()
end


-- Reset the game from the beginning.
function gameEnv:newGame()
    -- Start off with observations, but no actions or reward
    local data = self.NesEnv:envStart()
    data = data[1]

    return self:_updateState(data, 0, false):getState()
end

function gameEnv:nActions()
    return self.NesEnv.envSpec.nActions
end

--[[ Function returns the number total number of pixels in one frame/observation
from the current game.
]]
function gameEnv:nObsFeature()
    return torch.prod(torch.Tensor(
        self._state.observation[1]:size():totable()
    ),1)[1]
end

-- Human training functions
function gameEnv:fillTrainingActions()
    local training_actions = {}
    self.NesEnv:fillTrainingActions(training_actions, self._actrep)
    return training_actions
end

function gameEnv:fillAllTrainingActions()
    local training_actions = {}
    self.NesEnv:fillTrainingActions(training_actions, 1, 1, table.getn(self.NesEnv.trainingCache))
    return training_actions
end

