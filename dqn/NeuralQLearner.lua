if not dqn then
    require 'initenv'
end

local nql = torch.class('dqn.NeuralQLearner')


function nql:__init(args)
    self.state_dim  = args.state_dim -- State dimensionality.
    self.n_actions  = args.n_actions
    self.verbose    = args.verbose
    self.best       = args.best  -- Whether we should load the best or the latest network.

    --- epsilon annealing
    self.ep_start   = args.ep or 1
    self.ep         = self.ep_start -- Exploration probability.
    self.ep_end     = args.ep_end or self.ep
    self.ep_endt    = args.ep_endt or 1000000
    self.ep_human   = args.ep_human or 0

    ---- learning rate annealing
    self.lr_start       = args.lr or 0.01 --Learning rate.
    self.lr             = self.lr_start
    self.lr_end         = args.lr_end or self.lr
    self.lr_endt        = args.lr_endt or 1000000
    self.wc             = args.wc or 0  -- L2 weight cost.
    self.minibatch_size = args.minibatch_size or 1
    self.valid_size     = args.valid_size or 500

    --- Q-learning parameters
    self.discount       = args.discount or 0.99 --Discount factor.
    self.update_freq    = args.update_freq or 1
    -- Number of points to replay per learning step.
    self.n_replay       = args.n_replay or 1
    -- Number of steps after which learning starts.
    self.learn_start    = args.learn_start or 0
     -- Size of the transition table.
    self.replay_memory  = args.replay_memory or 1000000
    self.hist_len       = args.hist_len or 1
    self.rescale_r      = args.rescale_r
    self.max_reward     = args.max_reward
    self.min_reward     = args.min_reward
    self.clip_delta     = args.clip_delta
    self.target_q       = args.target_q
    self.bestq          = 0

    self.gpu            = args.gpu

    self.ncols          = args.ncols or 1  -- number of color channels in input
    self.input_dims     = args.input_dims or {self.hist_len*self.ncols, 84, 84}
    self.preproc        = args.preproc  -- name of preprocessing network
    self.histType       = args.histType or "linear"  -- history type to use
    self.histSpacing    = args.histSpacing or 1
    self.nonTermProb    = args.nonTermProb or 1
    self.nonEventProb   = args.nonEventProb
    self.bufferSize     = args.bufferSize or 512

    self.transition_params = args.transition_params or {}

    self.training_actions = {}

    self.network    = args.network
    assert(self.network, 'Network required')

    -- check whether there is a network file
    local network_function
    if not (type(self.network) == 'string') then
        error("The type of the network provided in NeuralQLearner" ..
              " is not a string!")
    end

    local msg, err = pcall(require, self.network)
    if not msg then
        local file_check, err_msg = lfs.attributes(self.network)
        if file_check then
            -- try to load saved agent
            print("Loading: ", self.network)
            local err_msg, exp = pcall(torch.load, self.network)

            if not err_msg then
                error("Could not find network file. Error: " .. exp)
            end
            if self.best and exp.best_model then
                print("Loading best model...")
                self.network = exp.best_model
            else
                print("Loading the latest (not necessarily the best) model...")
                self.network = exp.model
            end

            if self.verbose >= 2 then print(self.network) end

        else
            error("Network file not available: " .. err_msg)
        end
    else
        print('Creating Agent Network from ' .. self.network)
        self.network = err
        self.network = self:network()
    end

    if self.gpu and self.gpu >= 0 then
        self.network:cuda()
    else
        self.network:float()
    end

    -- Load preprocessing network.
    if not (type(self.preproc == 'string')) then
        error('The preprocessing is not a string')
    end
    msg, err = pcall(require, self.preproc)
    if not msg then
        error("Error loading preprocessing net. Error: " .. err)
    end
    self.preproc = err
    self.preproc = self:preproc()
    self.preproc:float()

    if self.gpu and self.gpu >= 0 then
        self.network:cuda()
        self.tensor_type = torch.CudaTensor
    else
        self.network:float()
        self.tensor_type = torch.FloatTensor
    end

    -- Create transition table.
    ---- assuming the transition table always gets floating point input
    ---- (Foat or Cuda tensors) and always returns one of the two, as required
    ---- internally it always uses ByteTensors for states, scaling and
    ---- converting accordingly
    local transition_args = {
        stateDim = self.state_dim, numActions = self.n_actions,
        histLen = self.hist_len, gpu = self.gpu,
        maxSize = self.replay_memory, histType = self.histType,
        histSpacing = self.histSpacing, nonTermProb = self.nonTermProb,
        bufferSize = self.bufferSize, nonEventProb = self.nonEventProb
    }

    self.transitions = dqn.TransitionTable(transition_args)

    self.numSteps = 0 -- Number of perceived states.
    self.lastState = nil
    self.lastAction = nil
    self.v_avg = 0 -- V running average.
    self.tderr_avg = 0 -- Temporal-difference error running average.

    self.q_max = 1
    self.r_max = math.max(torch.abs(self.max_reward), torch.abs(self.min_reward))

    self.w, self.dw = self.network:getParameters() -- Load the weights.
    self.dw:zero() -- Set gradient to zero.

    self.deltas = self.dw:clone():fill(0)

    self.tmp= self.dw:clone():fill(0)
    self.g  = self.dw:clone():fill(0)
    self.g2 = self.dw:clone():fill(0)

    -- Initialize the target nework to be equal to the current network.
    if self.target_q then
        self.target_network = self.network:clone()
    end
end


function nql:reset(state)
    if not state then
        return
    end
    self.best_network = state.best_network
    self.network = state.model
    self.w, self.dw = self.network:getParameters()
    self.dw:zero()
    self.numSteps = 0
end


function nql:preprocess(rawstate)
    if self.preproc then
      local input_state = self.preproc:forward(rawstate:float())
                                :clone():reshape(self.state_dim)
      return input_state
    end

    return rawstate
end

-- The idea here is to calculate the predicted ideal actions
-- each state in the minibatch, and to update the network
-- such that the prediction matches the actual desireable
-- outcome.
function nql:getQUpdate(args)
    local s, a, r, s2, term, delta
    local q, q2, q2_online, q2_target

    s = args.s
    a = args.a
    r = args.r
    s2 = args.s2
    term = args.term

    -- The order of calls to forward is a bit odd in order
    -- to avoid unnecessary calls (we only need 2).

    -- delta = r + (1-terminal) * gamma * avg_a Q(s2, a) - Q(s, a)

    -- XXX: These two "add dimension" operations needs unsqueeze, but torch7 doesn't have it

    -- term is 1D, but needs to be 2D for q2 cmul calculations
    term = torch.FloatTensor( 1, term:size(1) )
    term[1] = args.term:clone():float():mul(-1):add(1)
    term = term:t():expand( args.term:size(1), self.n_actions ):clone()

    -- a is 2D, but needs to be 3dish (and one-based) for some gather calls below
    a = torch.LongTensor( a:size(1), a:size(2), 1 )
    for i=1, a:size(1) do
        for j=1, a:size(2) do
            a[i][j][1] = args.a[i][j] + 1  -- 0 = Off = index 1, 1 = On = index 2
        end
    end

    -- If we don't have a target Q network yet, make one and use it.
    local target_q_net
    if self.target_q then
        target_q_net = self.target_network
    else
        target_q_net = self.network
    end

    -- Using *Double* DQN here for a multi-action selector
    -- For each s2 in the minibatch, pick the action with the highest value using the
    -- *online* network and then calculate the Q-value of s2 given this action using the
    -- *target* network.
    q2_online, best_a = self.network:forward(s2):float():max(3)  -- 3D -> 2Dish (Time x Action x 1 = Max Q)

    -- Get the Q-values for the best actions we identified above, using the *target* network.
    q2_target = target_q_net:forward(s2):float()       -- 3D (Time x Action x On/Off = Q)
    q2_max    = q2_target:gather(3, best_a):squeeze()  -- 2D (Time x Action = Max Q)

    -- Compute q2 = (1-terminal) * gamma * max_a Q(s2, a)
    -- Discounted by gamma and set to zero if terminal.
    -- 2D (Time X Action = Max Q*Gamma or 0)
    q2 = q2_max:clone():mul(self.discount):cmul(term)

    -- Set delta equal to the rewards in the minibatch.
    -- 1Dish -> 2D (Time X Action = Rewards)
    delta = torch.FloatTensor( 1, r:size(1) )
    delta[1] = r:clone():float()
    delta = delta:t():expand( r:size(1), self.n_actions ):clone()

    -- Rescale the reward to [-1, 1] if requested.
    if self.rescale_r then
        delta:div(self.r_max)
    end

    -- Add the discounted Q(s2, a) values to these rewards.
    delta:add(q2)  -- 2D + 2D

    -- q = Q(s,a)
    -- This estimates the value of state s for actions a using the *online* network,
    local q_all = self.network:forward(s):float()  -- 3D (Time x Action x On/Off = Q)
    q = q_all:gather(3, a):squeeze()               -- 2D (Time x Action = Q of each selected On/Off state)

    -- Finally, subtract out the Q(s, a) values.
    delta:add(-1, q)  -- 2D + 2D

    -- Keep the deltas bounded, if requested.
    if self.clip_delta then
        delta:clamp(-self.clip_delta, self.clip_delta)
    end

    -- Create targets equal to the deltas of each selected action
    -- 3D (Time x Action x On/Off = [after loop] deltas of each selected action or 0)
    local targets = torch.zeros(self.minibatch_size, self.n_actions, 2):float()
    for i=1,math.min(self.minibatch_size,a:size(1)) do
        for j=1,self.n_actions do
            targets[i][j][ a[i][j][1] ] = delta[i][j]
        end
    end

    if self.gpu >= 0 then targets = targets:cuda() end

    return targets, delta, q2_max
end

function nql:qLearnMinibatch()
    -- Perform a minibatch Q-learning update:
    -- w += lr * [r + (discount * max Q(s2,a2)) - Q(s,a)] * dQ(s,a)/dw
    assert(self.transitions:size() > self.minibatch_size)

    -- Load a minibatch of experiences.
    local s, a, r, s2, term = self.transitions:sample(self.minibatch_size)

    -- Feed these experiences into the Q network.
    local targets, delta, q2_max = self:getQUpdate{s=s, a=a, r=r, s2=s2, term=term}

    -- zero gradients of parameters
    self.dw:zero()

    -- Do a backwards pass to calculate the gradients.
    self.grad_input = self.network:backward(s, targets)

    -- add weight cost to gradient - this defaults to zero.
    self.dw:add(-self.wc, self.w)

    -- compute linearly annealed learning rate
    local t = math.max(0, self.numSteps - self.learn_start)
    self.lr = (self.lr_start - self.lr_end) * (self.lr_endt - t)/self.lr_endt +
                self.lr_end
    self.lr = math.max(self.lr, self.lr_end)

    -- use gradients
    self.g:mul(0.95):add(0.05, self.dw)
    self.tmp:cmul(self.dw, self.dw)
    self.g2:mul(0.95):add(0.05, self.tmp)
    self.tmp:cmul(self.g, self.g)
    self.tmp:mul(-1)
    self.tmp:add(self.g2)
    self.tmp:add(0.01)
    self.tmp:sqrt()

    -- accumulate update
    self.deltas:mul(0):addcdiv(self.lr, self.dw, self.tmp)
    self.w:add(self.deltas)
end

-- Returns a Tensor of input gradients
function nql:get_grad_input()
    if self.grad_input then
        local size = {}
        table.insert(size, self.minibatch_size)
        for _, dim in ipairs(self.input_dims) do table.insert(size, dim) end
        size = torch.LongStorage(size)

        -- 4D -> 2D (MBSize x Frames x X x Y --> X x Y)
        local gi     = self.grad_input:resize(size):float()
        local avg, _ = gi:mean(1)
        avg, _ = avg:mean(2)
        avg = avg:squeeze()

        return avg
    end

    return torch.FloatTensor({self.input_dims[2], self.input_dims[3]}):zeros()
end

-- Returns valid_size experiences as validation data.
function nql:sample_validation_data()
    local s, a, r, s2, term = self.transitions:sample(self.valid_size)
    self.valid_s    = s:clone()
    self.valid_a    = a:clone()
    self.valid_r    = r:clone()
    self.valid_s2   = s2:clone()
    self.valid_term = term:clone()
end

-- Compute the mean Q value and the TD error for our early validation experiences.
function nql:compute_validation_statistics()
    local targets, delta, q2_max = self:getQUpdate{s=self.valid_s,
        a=self.valid_a, r=self.valid_r, s2=self.valid_s2, term=self.valid_term}

    -- This is the average Q value of the target network for the highest-value action.
    -- This ideally should rise with learning and stabalize at a reasonable value...
    self.v_avg = self.q_max * q2_max:mean()

    -- This in essence is the difference between the target and current networks' value estimate for Q(s, a).
    -- This should approach zero with time as learning slows...
    self.tderr_avg = delta:clone():abs():mean()
end

-- Main function for observing the results and learning.
function nql:perceive(reward, rawstate, terminal, testing, testing_ep)
    -- Preprocess state (will be set to nil if terminal)
    local state = self:preprocess(rawstate):float()
    local curState

    -- Clip the reward to the max/min, if requested.
    if self.max_reward then
        reward = math.min(reward, self.max_reward)
    end
    if self.min_reward then
        reward = math.max(reward, self.min_reward)
    end

    -- Add the preprocessed state and terminal value to the recent state table.
    self.transitions:add_recent_state(state, terminal)

    -- Store transition s, a, r, s'
    if self.lastState and not testing then
        self.transitions:add(self.lastState, self.lastAction, reward,
                             self.lastTerminal, priority)
    end

    -- Load validation data once we're past the initial phase.
    -- This is just a sample of experiences.
    if self.numSteps == self.learn_start+1 and not testing then
        self:sample_validation_data()
    end

    -- Get the hist_len most recent frames...
    -- Dimensions should be (hist_len, width, height).
    curState= self.transitions:get_recent()

    -- Add a dimension to make this into a one-entry minibatch
    -- to keep the network happy.
    curState = curState:resize(1, unpack(self.input_dims))

    -- Use the Q network to select actions based on the trailing
    -- hist_len frames.
    local action = torch.ByteTensor(self.n_actions):fill(0)
    if not terminal then
        action = self:pickTrainingActions(testing_ep) or self:dipSwitchMatch(curState)
    end

    -- Add this action to our experiences.
    -- This makes the recent states list complete with frames and actions.
    self.transitions:add_recent_action(action)

    -- Learn...
    if self.numSteps > self.learn_start and not testing and
        self.numSteps % self.update_freq == 0 then
        for i = 1, self.n_replay do
            self:qLearnMinibatch()
        end
    end

    -- Track the number of learning steps we've undertaken.
    if not testing then
        self.numSteps = self.numSteps + 1
    end

    -- Save the state and action for the next round.
    self.lastState    = state:clone()
    self.lastAction   = action
    self.lastTerminal = terminal

    -- After target_q steps, replace the existing Q network with the newer one.
    if self.target_q and self.numSteps % self.target_q == 1 then
        -- If we're going to replace the large CUDA network, clean it out first
        self.target_network = nil
        collectgarbage()
        self.target_network = self.network:clone()
    end

    -- Return the action so we can feed it to the emulator.
    return action
end

-- Return an action for the given state.
function nql:pickTrainingActions(testing_ep)
    self.ep = testing_ep or (
        -- ep_start, ep_end = max/min Exploration Probability
        self.ep_end + math.max(0,
            (self.ep_start - self.ep_end) *  -- EP range (ie: 1-.1 = 90%)
            (self.ep_endt - math.max(0, self.numSteps - self.learn_start)) /
            self.ep_endt
        )
    )

    -- Select an action, maybe randomly.
    if torch.uniform() < self.ep then
        local btns = torch.ByteTensor(self.n_actions):fill(0)
        local in_human_training = testing_ep and testing_ep > 1

        if
            (in_human_training or torch.uniform() < self.ep_human) and
            self.training_actions and table.getn(self.training_actions) > 0
        then
            -- Select the top action from the human training pool
            local action = table.remove(self.training_actions, 1)
            for _, v in pairs(action) do
                btns[v] = 1
            end
        else
            -- Select multiple random actions, with probability ep.
            for i= 1, self.n_actions do
                -- Don't hit *all* the buttons at once.  Keep the mix at 25%.
                if torch.uniform() < 0.25 then btns[i] = 1 end
            end
        end

        return btns
    end

    return nil
end

-- Return the actions based on a 2D map of "dip switches", using the on/off state with
-- the highest Q value
function nql:dipSwitchMatch(state)

    -- Turn single state into minibatch.  Needed for convolutional nets.
    if state:dim() == 2 then
        assert(false, 'Input must be at least 3D')
        state = state:resize(1, state:size(1), state:size(2))
    end

    if self.gpu >= 0 then
        state = state:cuda()
    end

    -- Feed the state into the current network.
    local q = self.network:forward(state):float():squeeze()

    -- Mark the ones that are on (row 2), and use those actions
    local btns = torch.ByteTensor(self.n_actions):fill(0)
    for i = 1, self.n_actions do
        if q[i][2] > q[i][1] then btns[i] = 1 end
    end

    return btns
end


function nql:_loadNet()
    local net = self.network
    if self.gpu then
        net:cuda()
    else
        net:float()
    end
    return net
end


function nql:init(arg)
    self.n_actions = arg.n_actions
    self.network = self:_loadNet()
    -- Generate targets.
    self.transitions:empty()
end


function nql:report()
    print(get_weight_norms(self.network))
    print(get_grad_norms(self.network))
end

-- Prints the hist_len most recent frames.
-- Assumes images are square...
function nql:printRecent()

    print("Saving frame snapshot...")

    for i = 1, self.hist_len do

      local filename = "Frame" .. self.transitions.histIndices[i] .. ".png"
      image.save(filename, self.transitions.recent_s[self.transitions.histIndices[i]]:resize(self.ncols, self.state_dim^.5, self.state_dim^.5))
      --image.save(filename, self.transitions.recent_s[i]:clone():resize(self.ncols, self.state_dim^.5, self.state_dim^.5))

    end
end
