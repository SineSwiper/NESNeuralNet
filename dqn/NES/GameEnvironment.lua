-- TODO: Refactor and merge

-- This file defines the NES.GameEnvironment class.

-- The GameEnvironment class.
local gameEnv = torch.class('NES.GameEnvironment')


function gameEnv:__init(_opt)
    local _opt = _opt or {}
    -- defaults to emulator speed
    self.game_path      = _opt.game_path or '.'
    self.verbose        = _opt.verbose or 0
    self._actrep        = _opt.actrep or 1
    self._random_starts = _opt.random_starts or 1
    self.gameOverPenalty = _opt.gameOverPenalty or 0
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
    local env
    local params = _params or {useRGB=true}
    -- if no game name given use previous name if available
    if self.game then
        env = self.game.name
    end
    env = _env or env

    self.game = NES.game(env, params, self.game_path)
    self.game:resetGame()

    -- start the game
    if self.verbose > 0 then
        print('\nPlaying:', self.game.name)
    end

    self:_resetState()
    self:_updateState(self:_step({}))
    self:getState()
    return self
end


function gameEnv:_resetState()
    self._state = self._state or {}
    return self
end


-- Function plays `action` in the game and return game state.
function gameEnv:_step(action)
    assert(action, 'action is required')
    assert(type(action) == 'table', 'action needs to be a table')
    local x = self.game:play(action)
    return x.data, x.reward, x.terminal
end

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
        data, reward, terminal = self:_step(action)

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
    self.game:resetGame()
    -- take one null action in the new game
    return self:_updateState(self:_step({})):getState()
end


--[[ Function advances the emulator state until a new (random) game starts and
returns this state.
]]
function gameEnv:nextRandomGame(k)
    local obs, reward, terminal = self:newGame()
    k = k or torch.random(self._random_starts)
    for i=1,k-1 do
        obs, reward, terminal = self:_step(0)
        if terminal then
            print(string.format('WARNING: Terminal signal received after %d 0-steps', i))
        end
    end
    return self:_updateState(self:_step(0)):getState()
end

function gameEnv:nActions()
    return self.game.env.envSpec.nActions
end

--[[ Function returns the number total number of pixels in one frame/observation
from the current game.
]]
function gameEnv:nObsFeature()
    return self.game:nObsFeature()
end

