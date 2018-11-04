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
    self:reset(_opt.env, _opt.env_params)
    return self
end


function gameEnv:_updateState(frame, reward, terminal)
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

    self.game       = NES.game(env, params, self.game_path)
    self._actions   = self:getActions()

    -- start the game
    if self.verbose > 0 then
        print('\nPlaying:', self.game.name)
    end

    self:_resetState()
    self:_updateState(self:_step(0))
    self:getState()
    return self
end


function gameEnv:_resetState()
    self._state = self._state or {}
    return self
end


-- Function plays `action` in the game and return game state.
function gameEnv:_step(action)
    assert(action)
    local x = self.game:play(action)
    return x.data, x.reward, x.terminal
end


-- Function plays one random action in the game and return game state.
function gameEnv:_randomStep()
    return self:_step(self._actions[torch.random(#self._actions)])
end


function gameEnv:step(action, training)
    -- accumulate rewards over actrep action repeats
    local cumulated_reward = 0
    local frame, reward, terminal
    for i=1,self._actrep do
        -- Take selected action; actions start with action "0".
        frame, reward, terminal = self:_step(action)

        -- accumulate instantaneous reward
        cumulated_reward = cumulated_reward + reward

        -- game over, no point to repeat current action
        if terminal then break end
    end
    self:_updateState(frame, cumulated_reward, terminal)
    return self:getState()
end


-- Reset the game from the beginning.
function gameEnv:newGame()
    self.game:resetGame()
    -- take one null action in the new game
    return self:_updateState(self:_step(0)):getState()
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


--[[ Function returns the number total number of pixels in one frame/observation
from the current game.
]]
function gameEnv:nObsFeature()
    return self.game:nObsFeature()
end


-- Function returns a table with valid actions in the current game.
function gameEnv:getActions()
    return self.game:actions()
end
