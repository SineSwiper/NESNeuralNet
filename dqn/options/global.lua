-- options.global

require 'table'
require 'lfs'

-- The global ROOT_PATH
ROOT_PATH = '..'  -- Root path for all other paths (networks, movies, etc.)

local opt = {}
opt.pool_frms    = {}
opt.env_params   = {}
opt.agent_params = {}

-- GAME NAME
opt.gamename=string.match(rom.getfilename(), "^[%w _-&'%.!]+")
opt.gamename=string.gsub(opt.gamename, "[ '%.!]", '')

-- FRAMEWORK OPTIONS
opt.framework="NES"       -- New NES Lua Wrapper
opt.steps=5000000         -- Total steps to run the model

-- NES ENV OPTIONS
opt.env_params.useRGB=true
opt.env_params.useRAM=false  -- NN for RAM not supported yet... coming soon!

-- PREPROCESSOR OPTIONS
opt.agent_params.preproc="net_downsample_2x_full_y"
opt.agent_params.state_dim=7056      -- The number of pixels in the screen, with 84x84 frames
opt.agent_params.ncols=1             -- Represents just the Y (ie - grayscale) channel.
opt.initial_priority="false"

-- AGENT OPTIONS
opt.agent="NeuralQLearner"  -- Name of agent file to use
opt.agent_type="DQN3_0_3"

opt.agent_params.ep=1               -- The probability of choosing a random/human action rather than the best predicted action.
opt.agent_params.ep_end=0.01        -- What epsilon ends up as going forward.
opt.agent_params.ep_endt=1000000    -- This probability decreases over time, presumably as we get better.
opt.agent_params.ep_human=0.90      -- The probability (after EP is hit) that the action will be from the human training set.
opt.agent_params.max_reward=10000   -- Rewards are clipped to this value.
opt.agent_params.min_reward=-10000  -- Ditto.
opt.agent_params.rescale_r=1        -- Rescale rewards to [-1, 1]

opt.actrep=8            -- Number of times an action is repeated

-- LEARNING OPTIONS
opt.agent_params.lr=0.00025             -- Learning rate
opt.agent_params.bufferSize=3500        -- Size of the experience buffer
opt.agent_params.valid_size=1000        -- Size used for validation
opt.agent_params.learn_start=3500       -- Only start learning after this many steps. Should be bigger than bufferSize.
opt.agent_params.replay_memory=500000   -- Set small to speed up debugging.  Big memory object!
opt.agent_params.nonEventProb=nil       -- Probability of selecting a non-reward-bearing experience.
opt.agent_params.clip_delta=1           -- Limit the delta to +/- 1.

-- MINIBATCH OPTIONS
opt.agent_params.minibatch_size=3500    -- Size of each minibatch
opt.agent_params.n_replay=1             -- Minibatches to learn from each learning step.
opt.agent_params.update_freq=500        -- How often do we update the Q network?

-- Q NETWORK OPTIONS
opt.agent_params.network="convnet_nes"  -- Reload pretrained network
opt.agent_params.target_q=30000         -- Steps to replace target network with the updated one.
opt.agent_params.hist_len=4             -- Number of trailing frames to input into the Q network.
opt.agent_params.discount=0.99          -- Discount rate given to future rewards.

-- VALIDATION AND EVALUATION
opt.eval_freq=50000       -- Evaluate the model every eval_freq steps by calculating the score per episode for a few games.
opt.eval_steps=10000      -- How many steps does an evaluation last?
opt.prog_freq=50000       -- How often do you want a progress report?
opt.human_training=true   -- Play human training movies before learning with random steps?

-- PERFORMANCE AND DEBUG OPTIONS
opt.gpu=0            -- Zero means "use the GPU" which is a bit confusing... -1 for CPU.
opt.threads=16       -- Number of BLAS threads
opt.verbose=3        -- 2 is default. 3 turns on debugging messages about what the model is doing.
opt.seed=1           -- Fixed input seed for repeatable experiments
opt.best=1           -- Always load the best network

-- SAVE OPTIONS
opt.name=table.concat({opt.agent_type, opt.gamename, 'FULL_Y'}, "_")  -- Filename used for saving most things
opt.saveNetworkParams=true  -- Saves the agent network in a separate file
opt.save_versions=0         -- Append floor(step / opt.save_versions) to the filename
opt.save_freq=50000         -- Save every save_freq steps. Save early and often!

-- MOVIE OPTIONS
opt.example_mp=0.001        -- The probability of recording a random movie in the examples subdir

globalDQNOptions = opt
