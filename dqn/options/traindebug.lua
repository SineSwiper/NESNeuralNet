-- traindebug.lua

-- Used to get to the real Q network functions right away

require "options.global"

local opt = globalDQNOptions

opt.replay_memory=100
opt.agent_params.ep=1
opt.agent_params.ep_end=0.1
opt.agent_params.ep_endt=500
opt.agent_params.lr=0.01
opt.agent_params.target_q=10000
opt.agent_params.bufferSize=500
opt.agent_params.valid_size=450
opt.agent_params.learn_start=550

opt.eval_freq=500
opt.eval_steps=100
opt.prog_freq=500

opt.agent_type="TRAIN_DEBUG"
opt.name=table.concat({opt.agent_type, opt.env, 'FULL_Y'}, "_")

opt.actrep=4

-- SAVE OPTIONS
opt.network='../networks/DQN3_0_1_SuperMarioBros_FULL_Y.t7'
opt.saveNetworkParams=false
opt.save_versions=0
opt.save_freq=50000000


