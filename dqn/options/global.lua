-- options.global

require 'table'
require 'lfs'

local opt = {}
opt.pool_frms    = {}
opt.env_params   = {}
opt.agent_params = {}

-- ROM OPTIONS
opt.env="SuperMarioBros"

-- FRAMEWORK OPTIONS
opt.framework="NES"       -- New NES Lua Wrapper
opt.steps=5000000         -- Total steps to run the model
opt.game_path='../roms/'  -- Path to environment file (ROM)

-- NES ENV OPTIONS
opt.env_params.useRGB=true

-- PREPROCESSOR OPTIONS
opt.agent_params.preproc="net_downsample_2x_full_y"
opt.agent_params.state_dim=7056      -- The number of pixels in the screen, with 84x84 frames
opt.agent_params.ncols=1             -- Represents just the Y (ie - grayscale) channel.
opt.initial_priority="false"

-- AGENT OPTIONS
opt.agent="NeuralQLearner"  -- Name of agent file to use
opt.agent_type="DQN3_0_1"
opt.name=table.concat({opt.agent_type, opt.env, 'FULL_Y'}, "_")  -- Filename used for saving network and training history

opt.agent_params.ep=1               -- The probability of choosing a random action rather than the best predicted action.
opt.agent_params.ep_end=0.01        -- What epsilon ends up as going forward.
opt.agent_params.ep_endt=1000000    -- This probability decreases over time, presumably as we get better.
opt.agent_params.max_reward=10000   -- Rewards are clipped to this value.
opt.agent_params.min_reward=-10000  -- Ditto.
opt.agent_params.rescale_r=1        -- Rescale rewards to [-1, 1]
opt.agent_params.bufferSize=1024
opt.agent_params.valid_size=1000

opt.gameOverPenalty=1   -- Gives a negative reward upon dying.
opt.actrep=8            -- Number of times an action is repeated

-- LEARNING OPTIONS
opt.agent_params.lr=0.00025  -- .00025 for Atari.
opt.agent_params.learn_start=50000        -- Only start learning after this many steps. Should be bigger than bufferSize. Was set to 50k for Atari.
opt.agent_params.replay_memory=1000000    -- Set small to speed up debugging. 1M is the Atari setting... Big memory object!
opt.agent_params.n_replay=4               -- Minibatches to learn from each learning step.
opt.agent_params.nonEventProb=nil         -- Probability of selecting a non-reward-bearing experience.
opt.agent_params.clip_delta=1             -- Limit the delta to +/- 1.
opt.agent_params.minibatch_size=32

-- Q NETWORK OPTIONS
opt.agent_params.network="convnet_nes"  -- Reload pretrained network
opt.agent_params.target_q=30000         -- Steps to replace target nework with the updated one. Atari: 10k. DoubleDQN: 30k
opt.agent_params.update_freq=4          -- How often do we update the Q network?
opt.agent_params.hist_len=4             -- Number of trailing frames to input into the Q network. 4 for Atari...
opt.agent_params.discount=0.99          -- Discount rate given to future rewards.

-- VALIDATION AND EVALUATION
opt.eval_freq=50000     -- Evaluate the model every eval_freq steps by calculating the score per episode for a few games. 250k for Atari.
opt.eval_steps=10000    -- How many steps does an evaluation last? 125k for Atari.
opt.prog_freq=50000     -- How often do you want a progress report?

-- PERFORMANCE AND DEBUG OPTIONS
opt.gpu=-1           -- Zero means "use the GPU" which is a bit confusing... -1 for CPU.
opt.threads=8        -- Number of BLAS threads
opt.verbose=3        -- 2 is default. 3 turns on debugging messages about what the model is doing.
opt.random_starts=0  -- How many NOOPs to perform at the start of a game (random number up to this value). Shouldn't matter for SMB?
opt.seed=1           -- Fixed input seed for repeatable experiments

-- SAVE OPTIONS
opt.network='../networks/' .. opt.name .. '.t7'
opt.saveNetworkParams=true  -- Saves the agent network in a separate file
opt.save_versions=0         -- Append floor(step / opt.save_versions) to the filename
opt.save_freq=50000         -- Save every save_freq steps. Save early and often!

globalDQNOptions = opt
