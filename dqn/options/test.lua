-- test.lua

require "options.global"

-- TEST-SPECIFIC OPTIONS
globalDQNOptions.replay_memory=100             -- This doesn't matter for testing...
globalDQNOptions.agent_params.ep_end=0.1
globalDQNOptions.agent_params.ep_endt=500000
globalDQNOptions.agent_params.lr=0.01
globalDQNOptions.agent_params.target_q=10000
globalDQNOptions.agent_params.minibatch_size=256
globalDQNOptions.gif_file="../gifs/" .. globalDQNOptions.env .. ".gif"   -- GIF path to write session screens
globalDQNOptions.csv_file=""                                             -- CSV path to write session data

