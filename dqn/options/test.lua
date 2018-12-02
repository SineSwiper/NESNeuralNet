-- test.lua

require "options.global"
local opt = globalDQNOptions

-- TEST-SPECIFIC OPTIONS
opt.testing = true
opt.testing_ep = 0
opt.heatmap = true
opt.human_training = false
opt.replay_memory=100             -- This doesn't matter for testing...
opt.agent_params.ep_end=0.01
opt.agent_params.ep_endt=500
opt.agent_params.lr=0.000001
opt.agent_params.target_q=1000000

